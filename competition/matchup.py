"""
Match runner: drives a generals game between two stdio subprocess agents.

For each turn the runner:
  1. computes both observations from the env
  2. writes each observation frame to the corresponding subprocess's stdin
  3. reads one action line back from each subprocess's stdout
  4. steps the game with both actions, through whatever modifiers the ruleset
     switches on (see make_transition)

Each agent is a `run.sh` script (or compiled equivalent) speaking the
protocol in `competition/protocol.py`. The convention is to keep agents
under `competition/agents/<name>/`, but the runner doesn't care — pass
any path to a `run.sh`.

Usage:
    python competition/matchup.py [agent0_run.sh] [agent1_run.sh] [flags]
    python competition/matchup.py --mode competition      # pin the competition ruleset

Defaults to two copies of `competition/agents/expander_python/run.sh`.
"""
import argparse
import subprocess
import sys
from pathlib import Path

import jax.numpy as jnp
import jax.random as jrandom

from generals import GeneralsEnv
from generals.core import game
from generals.core.game import create_initial_state
from generals.core.grid import generate_grid
from generals.modifiers import build_castles as _bc
from generals.modifiers import deathtouch as _dt
from protocol import (
    decode_action,
    encode_handshake,
    encode_observation,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_AGENT = REPO_ROOT / "competition" / "agents" / "expander_python" / "run.sh"


def build_agent(run_sh: Path) -> None:
    """Run `build.sh` next to `run.sh` if one exists. Aborts on failure."""
    build = run_sh.parent / "build.sh"
    if not build.exists():
        return
    print(f"[matchup] building {build.relative_to(REPO_ROOT)}", file=sys.stderr)
    result = subprocess.run(["bash", str(build)], cwd=str(build.parent))
    if result.returncode != 0:
        sys.exit(f"[matchup] build failed: {build}")


def spawn_agent(run_sh: Path, player_id: int, H: int, W: int, label: str) -> subprocess.Popen:
    proc = subprocess.Popen(
        ["bash", str(run_sh)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=sys.stderr,
        bufsize=1,
        text=True,
        cwd=str(run_sh.parent),
    )
    proc.stdin.write(encode_handshake(player_id, H, W))
    proc.stdin.flush()
    try:
        rel = run_sh.relative_to(REPO_ROOT)
    except ValueError:
        rel = run_sh
    print(f"[matchup] spawned {label} as player {player_id} "
          f"(pid={proc.pid}, {rel})", file=sys.stderr)
    return proc


def ask_agent(proc: subprocess.Popen, obs) -> jnp.ndarray:
    proc.stdin.write(encode_observation(obs))
    proc.stdin.flush()
    line = proc.stdout.readline()
    if not line:
        raise RuntimeError(f"agent (pid={proc.pid}) closed stdout unexpectedly")
    return decode_action(line)


def close_agent(proc: subprocess.Popen) -> None:
    # The agent treats EOF on stdin as "game over, exit cleanly".
    try:
        proc.stdin.close()
    except (BrokenPipeError, ValueError):
        pass
    try:
        proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def make_board(env, seed):
    """Build ONE starting board for `env` from `seed`.

    env.reset() would generate a 10k-state training pool, and env.init_state()
    always hands back a square max_grid_size board padded to pad_to — for the
    "competition" ruleset that is a 22x22 shape the evaluator never plays. Under
    a variable-size ruleset, draw each side independently, as the evaluator does.
    """
    key = jrandom.PRNGKey(seed)
    if env._fixed_dims is not None:
        return env.init_state(key)

    kd, kg = jrandom.split(key)
    lo, hi = env.min_grid_size, env.max_grid_size
    h = int(jrandom.randint(kd, (), lo, hi + 1))
    w = int(jrandom.randint(jrandom.fold_in(kd, 1), (), lo, hi + 1))
    grid = generate_grid(
        kg, grid_dims=(h, w),
        mountain_density_range=env.mountain_density_range,
        num_castles_range=env.num_castles_range,
        # the ruleset's spawn floor, in walking steps around the mountains;
        # generate_grid pads bottom/right for pooling, so the slice below trims
        # it back to the exact rectangle.
        min_generals_distance=env.min_generals_distance,
        castle_val_range=env.castle_val_range,
    )[:h, :w]
    if env.build_castles:
        # nothing neutral to capture — every castle in the game gets built
        grid = _bc.strip_neutral_castles(grid)
    return create_initial_state(grid.astype(jnp.int32))


def make_transition(env):
    """Return the env's per-turn transition: ruleset modifiers + the base step.

    `game.step` is only the base game. The modifiers a ruleset switches on are
    applied *around* it, and skipping them silently changes the rules rather
    than erroring: `game.step` tests `pass == 1`, so a build action
    ([2, row, col, ...]) falls through to a plain move and marches the army out
    of the cell instead of building, and no deathtouch ever fires. This is the
    same composition env.step() performs, minus its vectorised-training pool.
    """
    def transition(state, actions):
        if env.build_castles:
            # builds resolve first and come back rewritten as passes
            state, actions = _bc.apply_build_actions(state, actions)
        if env.deathtouch_turn is not None:
            return _dt.step(state, actions, env.deathtouch_turn)
        return game.step(state, actions)
    return transition


def replay(states_log, infos_log, agent_ids, fps):
    """Open an interactive replay of a recorded match.

    Controls (also drawn in the window): SPACE play/pause; Left/Right (or H/L)
    step one frame, hold to run through frames; R restart; Q quit.
    """
    from generals.gui import ReplayGUI
    from generals.gui.properties import GuiMode

    gui = ReplayGUI(states_log[0], agent_ids=agent_ids, fps=fps,
                    mode=GuiMode.REPLAY, start_paused=True)
    print("[matchup] replay open — controls are shown in the window")
    gui.play(states_log, infos_log)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("agent0", nargs="?", default=str(DEFAULT_AGENT),
                        help="path to player 0's run.sh")
    parser.add_argument("agent1", nargs="?", default=str(DEFAULT_AGENT),
                        help="path to player 1's run.sh")
    parser.add_argument("--grid-size", type=int, default=10,
                        help="env grid size (default: 10)")
    parser.add_argument("--truncation", type=int, default=400,
                        help="max turns before draw (default: 400)")
    parser.add_argument("--seed", type=int, default=0,
                        help="env RNG seed (default: 0)")
    parser.add_argument("--gui", action="store_true",
                        help="record the match, then open an interactive replay window")
    parser.add_argument("--fps", type=int, default=8,
                        help="GUI frame rate when --gui is set (default: 8)")
    parser.add_argument("--perfect-info", action="store_true",
                        help="disable fog of war; agents see the whole board")
    parser.add_argument("--mode", type=str, default=None,
                        help="named ruleset preset (e.g. competition); pins the full "
                             "ruleset and overrides --grid-size/--truncation/--perfect-info")
    args = parser.parse_args()

    a0_path = Path(args.agent0).resolve()
    a1_path = Path(args.agent1).resolve()
    for p in (a0_path, a1_path):
        if not p.exists():
            sys.exit(f"agent script not found: {p}")

    build_agent(a0_path)
    if a1_path != a0_path:
        build_agent(a1_path)

    # A named mode pins the whole ruleset for a competition round; otherwise the
    # individual flags build the env.
    if args.mode is not None:
        env = GeneralsEnv(mode=args.mode)
    else:
        env = GeneralsEnv(grid_dims=(args.grid_size, args.grid_size),
                          truncation=args.truncation,
                          perfect_info=args.perfect_info)

    # Observation mode comes from the env so a --mode preset controls it.
    get_obs = game.get_full_observation if env.perfect_info else game.get_observation

    # One board, and a turn loop that drives the jitted transition directly.
    # env.reset()/env.step() wrap it in vectorised-training machinery (auto-reset
    # from a 10k-state pool) that a single stdio match neither needs nor wants —
    # that eager pool indexing is what made the loop slow. What env.step() does
    # that we must keep doing ourselves is apply the ruleset modifiers.
    state = make_board(env, args.seed)
    H, W = (int(d) for d in state.armies.shape)

    # With --gui we record every frame during a full-speed (headless) match,
    # then open an interactive replay afterwards. Recording the whole game first
    # decouples match speed from the GUI frame rate and lets you scrub freely.
    record = args.gui
    states_log = [state] if record else None
    infos_log = [game.get_info(state)] if record else None

    labels = [a0_path.parent.name, a1_path.parent.name]
    agents = [
        spawn_agent(a0_path, 0, H, W, labels[0]),
        spawn_agent(a1_path, 1, H, W, labels[1]),
    ]

    transition = make_transition(env)
    built = [0, 0]
    prev_castles = state.castles
    winner = -1
    turn = 0
    try:
        while turn < env.truncation:
            obs_0 = get_obs(state, 0)
            obs_1 = get_obs(state, 1)

            a_0 = ask_agent(agents[0], obs_0)
            a_1 = ask_agent(agents[1], obs_1)

            actions = jnp.stack([a_0, a_1])
            state, info = transition(state, actions)
            turn += 1

            # Castles only exist mid-game under build-castles, and a build that
            # was priced out is consumed as a pass — so report the ones that
            # land, or a bot saving up for a castle has no way to tell whether
            # its build was accepted.
            if env.build_castles:
                born = state.castles & ~prev_castles
                if bool(born.any()):
                    for pid in (0, 1):
                        for r, c in zip(*jnp.where(born & state.ownership[pid])):
                            built[pid] += 1
                            print(f"[matchup] turn {turn}: player {pid} ({labels[pid]}) "
                                  f"built a castle at ({int(r)}, {int(c)})", file=sys.stderr)
                    prev_castles = state.castles

            if record:
                states_log.append(state)
                infos_log.append(info)

            if bool(info.is_done):
                winner = int(info.winner)
                break
    finally:
        for proc in agents:
            close_agent(proc)

    if winner >= 0:
        print(f"[matchup] turn {turn}: player {winner} captured the enemy general")
    else:
        print(f"[matchup] turn {turn}: truncated at {env.truncation} turns (draw)")
    if env.build_castles:
        print(f"[matchup] castles built: {built[0]} ({labels[0]}) "
              f"vs {built[1]} ({labels[1]})")

    if record:
        replay(states_log, infos_log, agent_ids=labels, fps=args.fps)


if __name__ == "__main__":
    main()
