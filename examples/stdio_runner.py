"""
Match runner: drives a generals game between two stdio subprocess agents.

For each turn the runner:
  1. computes both observations from the env
  2. writes each observation frame to the corresponding subprocess's stdin
  3. reads one action line back from each subprocess's stdout
  4. steps the env with both actions

Each agent is a `run.sh` script (or compiled equivalent) speaking the
protocol in `generals/protocol.py`. See `starters/<lang>/` for examples.

Usage:
    python examples/stdio_runner.py [-h] [--gui] [--grid-size N] [--fps N]
                                    [agent0_run.sh] [agent1_run.sh]

Defaults to two copies of `starters/python/run.sh`.
"""
import argparse
import subprocess
import sys
from pathlib import Path

import jax.numpy as jnp
import jax.random as jrandom

from generals import GeneralsEnv
from generals.core import game
from generals.protocol import (
    decode_action,
    encode_handshake,
    encode_observation,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_AGENT = REPO_ROOT / "starters" / "python" / "run.sh"


def build_agent(run_sh: Path) -> None:
    """Run `build.sh` next to `run.sh` if one exists. Aborts on failure."""
    build = run_sh.parent / "build.sh"
    if not build.exists():
        return
    print(f"[runner] building {build.relative_to(REPO_ROOT)}", file=sys.stderr)
    result = subprocess.run(["bash", str(build)], cwd=str(build.parent))
    if result.returncode != 0:
        sys.exit(f"[runner] build failed: {build}")


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
    print(f"[runner] spawned {label} as player {player_id} "
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
                        help="open a pygame replay window")
    parser.add_argument("--fps", type=int, default=8,
                        help="GUI frame rate when --gui is set (default: 8)")
    parser.add_argument("--perfect-info", action="store_true",
                        help="disable fog of war; agents see the whole board")
    args = parser.parse_args()

    # Match the env's observation mode for the per-turn obs we send to agents.
    get_obs = game.get_full_observation if args.perfect_info else game.get_observation

    a0_path = Path(args.agent0).resolve()
    a1_path = Path(args.agent1).resolve()
    for p in (a0_path, a1_path):
        if not p.exists():
            sys.exit(f"agent script not found: {p}")

    build_agent(a0_path)
    if a1_path != a0_path:
        build_agent(a1_path)

    env = GeneralsEnv(grid_dims=(args.grid_size, args.grid_size),
                      truncation=args.truncation,
                      perfect_info=args.perfect_info)
    key = jrandom.PRNGKey(args.seed)
    pool, state = env.reset(key)
    H = W = env.pad_to

    gui = None
    if args.gui:
        from generals.gui import ReplayGUI
        gui = ReplayGUI(state, agent_ids=[a0_path.parent.name, a1_path.parent.name])

    agents = [
        spawn_agent(a0_path, 0, H, W, "agent-0"),
        spawn_agent(a1_path, 1, H, W, "agent-1"),
    ]

    winner = -1
    truncated = False
    turn = 0
    try:
        while turn < env.truncation:
            obs_0 = get_obs(state, 0)
            obs_1 = get_obs(state, 1)

            a_0 = ask_agent(agents[0], obs_0)
            a_1 = ask_agent(agents[1], obs_1)

            actions = jnp.stack([a_0, a_1])
            timestep, state = env.step(state, actions, pool)
            turn += 1

            if gui is not None:
                gui.update(state, timestep.info)
                gui.tick(fps=args.fps)

            if bool(timestep.terminated):
                winner = int(timestep.info.winner)
                break
            if bool(timestep.truncated):
                truncated = True
                break
    finally:
        for proc in agents:
            close_agent(proc)
        if gui is not None:
            gui.close()

    if truncated:
        print(f"[runner] turn {turn}: truncated (draw)")
    elif winner >= 0:
        print(f"[runner] turn {turn}: player {winner} captured the enemy general")
    else:
        print(f"[runner] turn {turn}: stopped without resolution")


if __name__ == "__main__":
    main()
