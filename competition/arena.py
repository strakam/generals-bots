"""
Head-to-head evaluation harness for agent iterations.

Plays every seed in both seat orders (so seat advantage cancels out) and
reports win/loss/draw plus average end-of-game land, army and castles.

Usage:
    python competition/arena.py my_bot2 my_bot            # names under agents/
    python competition/arena.py my_bot3 expander_python --seeds 12
    python competition/arena.py my_bot6 my_bot5 --seeds 100 --jobs 8   # big run
    python competition/arena.py my_bot2 my_bot --grid-size 10   # non-competition

Every seed produces a different board (dimensions, terrain and general
positions all derive from it), so a run over N seeds is N randomised maps —
but the same N maps every time, which is what makes two comparisons
comparable. Use --seed-start to draw a fresh, disjoint set.

Runs the same ruleset path as matchup2.py (modifiers applied, per-seed dims).
"""
import argparse
import math
import statistics
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "competition"))

import jax
import jax.random as jrandom
import numpy as np

from generals import GeneralsEnv
from generals.core import game
from generals.modifiers import build_castles as _build_castles
from generals.modifiers import deathtouch as _deathtouch
from matchup import close_agent
from matchup2 import Transcript, make_init_state
from protocol import encode_handshake, encode_observation

AGENTS = REPO_ROOT / "competition" / "agents"


def _hms(seconds: float) -> str:
    seconds = int(seconds)
    if seconds >= 3600:
        return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m"
    if seconds >= 60:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    return f"{seconds}s"


def make_fast_stepper(env):
    """One jitted call per turn instead of two.

    matchup2.make_stepper dispatches apply_build_actions and deathtouch.step
    separately; on a 22x22 board the work is trivial and the per-call dispatch
    dominates. Fusing them under a single jit halves that overhead. The config
    reads are Python-level, so they are baked in at trace time.
    """
    build, dt = env.build_castles, env.deathtouch_turn

    @jax.jit
    def step(state, actions):
        if build:
            state, actions = _build_castles.apply_build_actions(state, actions)
        if dt is not None:
            return _deathtouch.step(state, actions, dt)
        return game.step(state, actions)
    return step


@jax.jit
def both_observations(state):
    """Both players' observations in one dispatch."""
    return game.get_observation(state, 0), game.get_observation(state, 1)


def spawn(run_sh: Path, player_id: int, H: int, W: int):
    """Start an agent, silently — matchup's spawn logs a line per process,
    which is 2 lines per match and drowns the progress bar."""
    proc = subprocess.Popen(
        ["bash", str(run_sh)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        bufsize=1, text=True, cwd=str(run_sh.parent),
    )
    proc.stdin.write(encode_handshake(player_id, H, W))
    proc.stdin.flush()
    return proc


def ask(proc, frame):
    """Send a frame, read one action line back. Returns a list of 5 ints."""
    proc.stdin.write(frame)
    proc.stdin.flush()
    line = proc.stdout.readline()
    if not line:
        raise RuntimeError(f"agent (pid={proc.pid}) closed stdout unexpectedly")
    return [int(x) for x in line.split()]


def resolve(name: str) -> Path:
    """Accept an agent directory name or a path to a run.sh."""
    p = Path(name)
    if p.suffix == ".sh" and p.exists():
        return p.resolve()
    run = AGENTS / name / "run.sh"
    if not run.exists():
        sys.exit(f"no agent {name!r} (looked for {run})")
    return run.resolve()


_WORKER = {}


def _init_worker(mode, grid_size, truncation):
    """Build one env per worker process (JAX import is the expensive part)."""
    if grid_size:
        env = GeneralsEnv(grid_dims=(grid_size, grid_size), truncation=truncation)
    else:
        env = GeneralsEnv(mode=mode)
    _WORKER["env"] = env
    _WORKER["step"] = make_fast_stepper(env)
    _WORKER["log"] = Transcript(None, False)


def _run_one(task):
    seed, path_a, path_b, swap, save_dir, meta_extra = task
    winner, turn, stats = play(_WORKER["env"], _WORKER["step"], _WORKER["log"],
                               seed, path_a, path_b, swap, save_dir, meta_extra)
    return seed, swap, winner, turn, stats


def play(env, step, log, seed, path_a, path_b, swap, save_dir=None, meta_extra=None):
    """One match. Returns (winner_label, turns, stats_by_label)."""
    labels = ["a", "b"]
    paths = [path_a, path_b]
    if swap:
        labels, paths = labels[::-1], paths[::-1]

    state, dims = make_init_state(env, jrandom.PRNGKey(seed))
    H = W = env.pad_to
    agents = [spawn(paths[i], i, H, W) for i in (0, 1)]

    winner = None
    turn = 0
    actions_log = [] if save_dir else None
    try:
        while turn < env.truncation:
            obs = both_observations(state)
            # One numpy array beats three small device transfers per turn.
            acts = np.array([ask(agents[i], encode_observation(obs[i])) for i in (0, 1)],
                            dtype=np.int32)
            if actions_log is not None:
                actions_log.append(acts)
            state, info = step(state, acts)
            turn += 1
            if bool(info.is_done):
                w = int(info.winner)
                winner = labels[w] if w >= 0 else None  # -1 = mutual deathtouch
                break
    finally:
        for p in agents:
            close_agent(p)

    stats = {}
    for i in (0, 1):
        o = game.get_observation(state, i)
        castles = int((np.asarray(state.castles, bool)
                       & np.asarray(state.ownership[i], bool)).sum())
        stats[labels[i]] = (int(o.owned_land_count), int(o.owned_army_count), castles)

    if actions_log:
        import replay as replay_io
        meta = dict(meta_extra or {})
        meta.update({
            "seed": seed,
            "board": f"{dims[0]}x{dims[1]}",
            "agents": [paths[i].parent.name for i in (0, 1)],
            "result": (f"turn {turn}: {'draw' if winner is None else winner + ' won'}"),
        })
        a_name, b_name = path_a.parent.name, path_b.parent.name
        seat = "p1" if swap else "p0"
        out = Path(save_dir) / f"{a_name}_vs_{b_name}_seed{seed:04d}_{seat}.rpl"
        replay_io.save(out, meta, actions_log)

    return winner, turn, stats


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("agent_a")
    ap.add_argument("agent_b")
    ap.add_argument("--seeds", type=int, default=8, help="number of seeds (default: 8)")
    ap.add_argument("--seed-start", type=int, default=0,
                    help="first seed (default: 0); use a different start for a "
                         "fresh set of maps")
    ap.add_argument("--jobs", type=int, default=1,
                    help="parallel worker processes (default: 1)")
    ap.add_argument("--mode", type=str, default="competition")
    ap.add_argument("--grid-size", type=int, default=None,
                    help="use a plain NxN board instead of --mode")
    ap.add_argument("--truncation", type=int, default=400)
    ap.add_argument("--save-dir", type=str, default=None, metavar="DIR",
                    help="write a replay file per match (~2 KB each) for later "
                         "viewing with competition/replay.py")
    ap.add_argument("--verbose", action="store_true",
                    help="print a line per match instead of the progress bar")
    ap.add_argument("--quiet", action="store_true",
                    help="no progress bar; totals only")
    args = ap.parse_args()

    path_a, path_b = resolve(args.agent_a), resolve(args.agent_b)
    name_a, name_b = path_a.parent.name, path_b.parent.name

    seeds = range(args.seed_start, args.seed_start + args.seeds)
    if args.save_dir:
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    meta_extra = {"mode": args.mode if not args.grid_size else None,
                  "grid_size": args.grid_size,
                  "truncation": args.truncation}
    tasks = [(seed, path_a, path_b, swap, args.save_dir, meta_extra)
             for seed in seeds for swap in (False, True)]

    wins = {"a": 0, "b": 0}
    draws = 0
    turns = []
    agg = {"a": [], "b": []}

    def progress(done, total, started):
        """One rewriting status line: bar, counts so far, elapsed and ETA."""
        elapsed = time.time() - started
        eta = (elapsed / done) * (total - done) if done else 0.0
        filled = int(28 * done / total)
        bar = "#" * filled + "-" * (28 - filled)
        sys.stderr.write(
            f"\r[{bar}] {done}/{total}  "
            f"{name_a} {wins['a']}W  {name_b} {wins['b']}W  {draws}D  "
            f"{_hms(elapsed)} elapsed, {_hms(eta)} left  ")
        sys.stderr.flush()

    def record(seed, swap, winner, turn, stats):
        nonlocal draws
        if winner is None:
            draws += 1
        else:
            wins[winner] += 1
        turns.append(turn)
        for k in ("a", "b"):
            agg[k].append(stats[k])
        if args.verbose:
            who = {"a": name_a, "b": name_b}.get(winner, "draw")
            print(f"seed {seed:>3} ({name_a} as {'p1' if swap else 'p0'}): "
                  f"{who:>18} on turn {turn}", flush=True)

    started = time.time()
    init_args = (args.mode, args.grid_size, args.truncation)
    # A \r-rewritten bar is noise when redirected to a file, so only draw it
    # for an actual terminal.
    show_bar = not (args.quiet or args.verbose) and sys.stderr.isatty()
    done = 0
    if args.jobs > 1:
        # Each worker pays the JAX import once, then plays matches back to back.
        with ProcessPoolExecutor(max_workers=args.jobs, initializer=_init_worker,
                                 initargs=init_args) as pool:
            for result in pool.map(_run_one, tasks):
                record(*result)
                done += 1
                if show_bar:
                    progress(done, len(tasks), started)
    else:
        _init_worker(*init_args)
        for task in tasks:
            record(*_run_one(task))
            done += 1
            if show_bar:
                progress(done, len(tasks), started)
    if show_bar:
        sys.stderr.write("\n")

    n = len(tasks)
    elapsed = time.time() - started
    board = args.mode if not args.grid_size else f"{args.grid_size}x{args.grid_size}"
    print(f"\n=== {name_a} vs {name_b} — {n} matches, {board}, "
          f"seeds {seeds.start}..{seeds.stop - 1}, {elapsed:.0f}s ===")
    print(f"{name_a}: {wins['a']}W   {name_b}: {wins['b']}W   draws: {draws}")

    # Score a draw as half a win (standard for match play) and put a 95%
    # interval on it, so "is this difference real?" has an answer.
    score = (wins["a"] + 0.5 * draws) / n
    margin = 1.96 * math.sqrt(max(score * (1 - score), 1e-9) / n)
    print(f"{name_a} score: {score:.3f} ± {margin:.3f}  "
          f"(95% CI {max(0, score - margin):.3f}–{min(1, score + margin):.3f}; "
          f"0.5 = even)")
    print(f"median game length: {int(statistics.median(turns))} turns")
    for key, name in (("a", name_a), ("b", name_b)):
        land = statistics.mean(s[0] for s in agg[key])
        army = statistics.mean(s[1] for s in agg[key])
        cast = statistics.mean(s[2] for s in agg[key])
        print(f"{name:>18}  avg final land {land:6.1f}  army {army:7.1f}  castles {cast:4.1f}")


if __name__ == "__main__":
    main()
