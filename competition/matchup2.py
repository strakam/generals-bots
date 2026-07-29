"""
Match runner with transcript logging — a drop-in wrapper around `matchup.py`.

Same flags and same behaviour as `matchup.py`, plus:

    --log PATH        write a transcript of the engine <-> agent conversation
    --show-grids      include the full observation frames (the three HxW grids)
                      in the transcript; off by default because it is ~2 MB per
                      400-turn match and the action lines are what you usually
                      want to read

Nothing is logged unless --log is passed, so the default cost is zero.

It also fixes two ruleset-fidelity gaps in `matchup.py`, so `--mode competition`
here actually plays the competition ruleset:

  * build-castles and deathtouch are applied (see make_stepper). matchup.py
    drives game.step directly, which skips both — and makes a `2 r c 0 0` build
    silently execute as a move upward rather than a no-op.
  * board dims are drawn per seed in [min_grid_size, max_grid_size] instead of
    always max x max (see make_init_state).

Both are no-ops outside a mode preset, so plain flag-driven matches behave
exactly as they did before.

Usage:
    python competition/matchup2.py [agent0_run.sh] [agent1_run.sh] [flags]
    python competition/matchup2.py --log match.log
    python competition/matchup2.py --log match.log --show-grids

Transcript format (line-oriented, greppable):

    # header: seed / grid / truncation / agent paths / engine module
    == handshake -> p0 (my_bot)
    0 20 20
    -- turn 0 -> p0 (my_bot)          [frame body only with --show-grids]
    <- turn 0 p0 (my_bot): 0 3 4 1 0    (2.1 ms)
    ...
    # result: ...

`<-` lines are agent replies, `->` lines are engine sends.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

# `python competition/matchup2.py` puts competition/ on sys.path but NOT the repo
# root, so a `generals` installed elsewhere (e.g. an editable install pointing at
# another checkout) would shadow this repo's engine and reject flags like
# perfect_info. Pin the repo root ahead of site-packages before importing.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import jax.numpy as jnp
import jax.random as jrandom

import generals
from generals import GeneralsEnv
from generals.core import game
from generals.modifiers import build_castles as _build_castles
from generals.modifiers import deathtouch as _deathtouch
from protocol import (
    decode_action,
    encode_handshake,
    encode_observation,
)
# Reuse the pieces that need no logging hooks, so this file stays a thin layer.
from matchup import DEFAULT_AGENT, build_agent, close_agent


def replay(states_log, infos_log, agent_ids, fps, cell_size=None):
    """Open an interactive replay. Same as matchup.replay, plus cell sizing.

    Controls (also drawn in the window): SPACE play/pause; Left/Right (or H/L)
    step one frame, hold to run through frames; R restart; Q quit.
    """
    from generals.gui import ReplayGUI
    from generals.gui.properties import GuiMode

    gui = ReplayGUI(states_log[0], agent_ids=agent_ids, fps=fps,
                    mode=GuiMode.REPLAY, start_paused=True, cell_size=cell_size)
    print(f"[matchup] replay open at {gui.cell_size}px cells — "
          "controls are shown in the window")
    gui.play(states_log, infos_log)


def make_stepper(env: GeneralsEnv):
    """Return a `step(state, actions) -> (state, info)` honouring env's modifiers.

    `matchup.py` calls `game.step` directly to skip GeneralsEnv.step's
    vectorised auto-reset machinery — but that also skips the build-castles and
    deathtouch modifiers, which live in GeneralsEnv.step and are part of the
    competition ruleset. Under raw game.step a build action is not even a no-op:
    execute_action only treats pass-field 1 as a pass, so `2 r c 0 0` falls
    through to _execute_move and silently moves the army upward.

    This mirrors the modifier half of GeneralsEnv.step and nothing else.
    """
    def step(state, actions):
        # Builds resolve before either player's move, then are rewritten to
        # passes so the base game never sees pass-field 2.
        if env.build_castles:
            state, actions = _build_castles.apply_build_actions(state, actions)
        if env.deathtouch_turn is not None:
            return _deathtouch.step(state, actions, env.deathtouch_turn)
        return game.step(state, actions)
    return step


def make_init_state(env: GeneralsEnv, key):
    """Build the opening state, sampling per-seed dims in variable-size modes.

    env.init_state() always generates max_grid_size x max_grid_size, so a
    competition match would be 21x21 every time. The real ruleset draws each
    side independently in [min_grid_size, max_grid_size] per game, and the eval
    driver scales the generals' minimum separation to 0.8 * min(h, w). Returns
    (state, (h, w)); either way the state is padded out to env.pad_to.
    """
    if env._fixed_dims is not None:
        return env.init_state(key), env._fixed_dims

    dim_key, state_key = jrandom.split(key)
    h, w = (int(x) for x in jrandom.randint(
        dim_key, (2,), env.min_grid_size, env.max_grid_size + 1))

    # _make_single_state_fixed reads min_generals_distance off the env; the
    # preset's value is a pool-wide floor, so scale it to these exact dims.
    original = env.min_generals_distance
    env.min_generals_distance = max(original, int(0.8 * min(h, w)))
    try:
        state = env._make_single_state_fixed(state_key, h, w)
    finally:
        env.min_generals_distance = original
    return state, (h, w)


class Transcript:
    """Append-only log of the wire conversation. A no-op when path is None."""

    def __init__(self, path: str | None, show_grids: bool):
        self.file = open(path, "w") if path else None
        self.show_grids = show_grids

    @property
    def enabled(self) -> bool:
        return self.file is not None

    def comment(self, text: str) -> None:
        if self.file:
            self.file.write(f"# {text}\n")

    def handshake(self, pid: int, label: str, text: str) -> None:
        if self.file:
            self.file.write(f"== handshake -> p{pid} ({label})\n{text}")

    def frame(self, turn: int, pid: int, label: str, text: str) -> None:
        if not self.file:
            return
        self.file.write(f"-- turn {turn} -> p{pid} ({label})\n")
        if self.show_grids:
            self.file.write(text)

    def action(self, turn: int, pid: int, label: str, line: str, ms: float) -> None:
        if self.file:
            self.file.write(f"<- turn {turn} p{pid} ({label}): {line.strip()}"
                            f"    ({ms:.1f} ms)\n")
            self.file.flush()   # so a hung match still leaves a usable tail

    def close(self) -> None:
        if self.file:
            self.file.close()
            self.file = None


def spawn_agent(run_sh: Path, player_id: int, H: int, W: int, label: str,
                log: Transcript) -> subprocess.Popen:
    """Spawn via matchup.spawn_agent, then record the handshake it sent."""
    import matchup
    proc = matchup.spawn_agent(run_sh, player_id, H, W, label)
    log.handshake(player_id, label, encode_handshake(player_id, H, W))
    return proc


def ask_agent(proc: subprocess.Popen, obs, turn: int, pid: int, label: str,
              log: Transcript) -> jnp.ndarray:
    """One request/response round trip, logged (and timed) if logging is on."""
    frame = encode_observation(obs)
    log.frame(turn, pid, label, frame)

    proc.stdin.write(frame)
    proc.stdin.flush()

    start = time.perf_counter()
    line = proc.stdout.readline()
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    if not line:
        log.comment(f"agent p{pid} ({label}) closed stdout at turn {turn}")
        log.close()
        raise RuntimeError(f"agent (pid={proc.pid}) closed stdout unexpectedly")

    log.action(turn, pid, label, line, elapsed_ms)
    return decode_action(line)


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
    parser.add_argument("--save", type=str, default=None, metavar="PATH",
                        help="record the match to a compact replay file "
                             "(watch it later with competition/replay.py)")
    parser.add_argument("--cell-size", type=int, default=None, metavar="PX",
                        help="pixels per board cell in --gui (default: auto-fit "
                             "the window to your screen)")
    parser.add_argument("--log", type=str, default=None, metavar="PATH",
                        help="write a transcript of the engine <-> agent conversation")
    parser.add_argument("--show-grids", action="store_true",
                        help="include full observation grids in --log (large)")
    args = parser.parse_args()

    if args.show_grids and not args.log:
        parser.error("--show-grids requires --log")

    a0_path = Path(args.agent0).resolve()
    a1_path = Path(args.agent1).resolve()
    for p in (a0_path, a1_path):
        if not p.exists():
            sys.exit(f"agent script not found: {p}")

    build_agent(a0_path)
    if a1_path != a0_path:
        build_agent(a1_path)

    if args.mode is not None:
        env = GeneralsEnv(mode=args.mode)
    else:
        env = GeneralsEnv(grid_dims=(args.grid_size, args.grid_size),
                          truncation=args.truncation,
                          perfect_info=args.perfect_info)

    get_obs = game.get_full_observation if env.perfect_info else game.get_observation

    step = make_stepper(env)

    key = jrandom.PRNGKey(args.seed)
    state, (board_h, board_w) = make_init_state(env, key)
    H = W = env.pad_to

    record = args.gui
    states_log = [state] if record else None
    infos_log = [game.get_info(state)] if record else None
    # Replays store actions only — the states are recomputed from the seed.
    actions_log = [] if args.save else None

    labels = [a0_path.parent.name, a1_path.parent.name]

    log = Transcript(args.log, args.show_grids)
    log.comment(f"seed={args.seed} board={board_h}x{board_w} pad_to={env.pad_to} "
                f"truncation={env.truncation} perfect_info={env.perfect_info} "
                f"mode={args.mode}")
    log.comment(f"build_castles={env.build_castles} deathtouch={env.deathtouch_turn}")
    log.comment(f"p0={a0_path}")
    log.comment(f"p1={a1_path}")
    log.comment(f"engine={generals.__file__}")
    log.comment(f"grids={'full' if args.show_grids else 'omitted (use --show-grids)'}")

    agents = [
        spawn_agent(a0_path, 0, H, W, labels[0], log),
        spawn_agent(a1_path, 1, H, W, labels[1], log),
    ]

    winner = -1
    turn = 0
    try:
        while turn < env.truncation:
            obs_0 = get_obs(state, 0)
            obs_1 = get_obs(state, 1)

            a_0 = ask_agent(agents[0], obs_0, turn, 0, labels[0], log)
            a_1 = ask_agent(agents[1], obs_1, turn, 1, labels[1], log)

            actions = jnp.stack([a_0, a_1])
            if actions_log is not None:
                actions_log.append(actions)
            state, info = step(state, actions)
            turn += 1

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
        result = f"turn {turn}: player {winner} captured the enemy general"
    else:
        result = f"turn {turn}: truncated at {env.truncation} turns (draw)"
    print(f"[matchup] {result}")

    log.comment(f"result: {result}")
    log.close()
    if args.log:
        print(f"[matchup] transcript written to {args.log}", file=sys.stderr)

    if args.save:
        import replay as replay_io
        meta = {
            "seed": args.seed,
            "mode": args.mode,
            "grid_size": args.grid_size,
            "truncation": env.truncation,
            "perfect_info": env.perfect_info,
            "board": f"{board_h}x{board_w}",
            "agents": labels,
            "result": result,
        }
        size = replay_io.save(args.save, meta, actions_log)
        print(f"[matchup] replay saved to {args.save} ({size / 1024:.1f} KB, "
              f"{len(actions_log)} turns)", file=sys.stderr)

    if record:
        replay(states_log, infos_log, agent_ids=labels, fps=args.fps,
               cell_size=args.cell_size)


if __name__ == "__main__":
    main()
