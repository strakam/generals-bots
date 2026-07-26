"""
Compact match recording and playback.

A match is fully determined by its ruleset, its seed and the two action
streams — the board, the general positions and every subsequent state are
recomputed from those. So a recording stores only the actions, and replaying
means re-simulating. A 1200-turn competition match is ~10 KB gzipped, versus
tens of megabytes for the states it expands into.

Record:
    python competition/matchup2.py a/run.sh b/run.sh --mode competition --save m.rpl

Watch:
    python competition/replay.py m.rpl
    python competition/replay.py m.rpl --fps 20 --cell-size 24

Inspect without opening a window:
    python competition/replay.py m.rpl --info

Format: gzipped JSON. `meta` pins the ruleset so playback reconstructs the
exact same game; `actions` is a flat list of 10 ints per turn (both players'
[pass, row, col, dir, split], player 0 first).
"""
import argparse
import gzip
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "competition"))

import numpy as np

FORMAT_VERSION = 1


def dumps(meta: dict, actions) -> bytes:
    """Serialise a recording to .rpl bytes. `actions` is a seq of (2,5) arrays."""
    flat = [int(v) for step in actions for row in np.asarray(step) for v in row]
    blob = {"version": FORMAT_VERSION, "meta": meta, "actions": flat}
    return gzip.compress(json.dumps(blob, separators=(",", ":")).encode(), 6)


def save(path, meta: dict, actions) -> int:
    """Write a recording. `actions` is a sequence of (2, 5) int arrays."""
    path = Path(path)
    path.write_bytes(dumps(meta, actions))
    return path.stat().st_size


def load(path):
    """Read a recording back as (meta, list of (2,5) int arrays)."""
    with gzip.open(path, "rb") as fh:
        blob = json.loads(fh.read().decode())
    if blob.get("version") != FORMAT_VERSION:
        sys.exit(f"{path}: unsupported replay version {blob.get('version')}")
    flat = blob["actions"]
    actions = [np.array(flat[i:i + 10], dtype=np.int32).reshape(2, 5)
               for i in range(0, len(flat), 10)]
    return blob["meta"], actions


def rebuild(meta, actions):
    """Re-simulate the match, returning (states, infos) for the GUI."""
    import jax.random as jrandom

    from generals import GeneralsEnv
    from generals.core import game
    from matchup2 import make_init_state, make_stepper

    if meta.get("mode"):
        env = GeneralsEnv(mode=meta["mode"])
    else:
        env = GeneralsEnv(grid_dims=(meta["grid_size"], meta["grid_size"]),
                          truncation=meta["truncation"],
                          perfect_info=meta.get("perfect_info", False))
    step = make_stepper(env)

    state, _ = make_init_state(env, jrandom.PRNGKey(meta["seed"]))
    states = [state]
    infos = [game.get_info(state)]
    for act in actions:
        state, info = step(state, act)
        states.append(state)
        infos.append(info)
    return env, states, infos


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="recording written by matchup2.py --save")
    ap.add_argument("--fps", type=int, default=8, help="playback rate (default: 8)")
    ap.add_argument("--cell-size", type=int, default=None, metavar="PX",
                    help="pixels per cell (default: auto-fit to screen)")
    ap.add_argument("--info", action="store_true",
                    help="print the header and exit, no window")
    args = ap.parse_args()

    meta, actions = load(args.path)
    size = Path(args.path).stat().st_size
    print(f"[replay] {args.path}  {size / 1024:.1f} KB  {len(actions)} turns")
    print(f"[replay] {meta.get('agents')}  seed={meta['seed']} "
          f"mode={meta.get('mode')} board={meta.get('board')}")
    print(f"[replay] result: {meta.get('result')}")
    if args.info:
        return

    env, states, infos = rebuild(meta, actions)
    from generals.gui import ReplayGUI
    from generals.gui.properties import GuiMode

    names = meta.get("agents") or ["Player 0", "Player 1"]
    gui = ReplayGUI(states[0], agent_ids=list(names), fps=args.fps,
                    mode=GuiMode.REPLAY, start_paused=True,
                    cell_size=args.cell_size)
    print(f"[replay] window open at {gui.cell_size}px cells — "
          "controls are shown in the window")
    gui.play(states, infos)


if __name__ == "__main__":
    main()
