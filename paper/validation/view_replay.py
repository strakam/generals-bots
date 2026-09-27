#!/usr/bin/env python
"""Watch a generals.io game in the simulator's GUI, by replay id or .gior file.

    python paper/validation/view_replay.py C-sINIgzc            # downloads C-sINIgzc.gior if needed, replays and shows it
    python paper/validation/view_replay.py path/to/game.gior --fps 20
    python paper/validation/view_replay.py C-sINIgzc --dir agent_replays/gior   # look in a local archive first

The recorded moves are fed through the engine turn by turn (with the general-trade rule on, and the
site's leave events applied where the engine provides them), so what is shown is the engine's own
board, which agrees with the site's engine on every cell of every turn on the validated replays.
Controls: SPACE play/pause, Left/Right step (hold to run), R restart, Q quit.
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import argparse, sys, time, urllib.parse, urllib.request  # noqa: E402
import numpy as np  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "ffa"))
import gior  # noqa: E402
from board_diff_ffa import prepare  # noqa: E402  (N players, teams, leave events)
import jax.numpy as jnp  # noqa: E402
from generals.core import game  # noqa: E402

BUCKETS = ("https://generalsio-replays-na.s3.amazonaws.com/", "https://generalsio-replays-eu.s3.amazonaws.com/",
           "https://generalsio-replays-bot.s3.amazonaws.com/")


def fetch(replay_id: str, cache_dir: str, dirs: list[str]) -> str:
    """The replay file: from one of `dirs` (e.g. a local archive of the agent's games), the cache, or the site."""
    for d in [*dirs, cache_dir]:
        p = os.path.join(d, replay_id + ".gior")
        if os.path.exists(p): return p
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, replay_id + ".gior")
    for base in BUCKETS:
        try:
            data = urllib.request.urlopen(urllib.request.Request(base + urllib.parse.quote(replay_id) + ".gior",
                                                                 headers={"User-Agent": "Mozilla/5.0"}), timeout=30).read()
            open(path, "wb").write(data); return path
        except urllib.error.HTTPError as e:
            if e.code != 404: raise
    sys.exit(f"replay {replay_id} not found in the public replay buckets")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("replay", help="replay id (e.g. C-sINIgzc) or path to a .gior file")
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--cache", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "replays"))
    ap.add_argument("--dir", action="append", default=[], help="directory of .gior files to look in first (repeatable); "
                    "the GENERALS_REPLAY_DIRS environment variable (colon-separated) adds more")
    ap.add_argument("--paused", action="store_true", help="start paused")
    ap.add_argument("--no-gui", action="store_true", help="only replay and print the outcome")
    a = ap.parse_args()
    dirs = a.dir + [d for d in os.environ.get("GENERALS_REPLAY_DIRS", "").split(":") if d]
    path = a.replay if a.replay.endswith(".gior") else fetch(a.replay, a.cache, dirs)
    row = gior.row(path); prep = prepare(row)
    if prep["exclude"]: sys.exit(f"cannot replay {row['id']}: {prep['exclude']}")
    names = row["usernames"]; N = len(names); W, H = row["mapWidth"], row["mapHeight"]
    grid = prep["grid"]; acts = prep["actions"]; afk = prep["afk"]; teams = prep["teams"]
    n_pad = acts.shape[1]
    print(f"{row['id']}: {N} players {names}, map {W}x{H}, {len(row['moves'])} moves, "
          f"{acts.shape[0]} turns, format {row['version']}" + (f", teams {[int(x) for x in teams[:N]]}" if row.get('teams') else ""))
    state = game.create_initial_state(jnp.asarray(grid), teams=jnp.asarray(teams))
    has_events = hasattr(game, "surrender") and hasattr(game, "neutralize")
    if afk.any() and not has_events:
        print("note: this engine has no leave events; the board after a player leaves may differ from the site")

    # replay the whole game first (fast), then hand the frames to the GUI's interactive replay loop
    states, infos = [state], [game.get_info(state)]
    end = None
    for t in range(acts.shape[0]):
        if has_events:
            for p in range(N):
                for _ in range(int(afk[t, p])):
                    state = game.neutralize(state, p) if bool(state.eliminated[p]) else game.surrender(state, p)
        state, info = game.step(state, jnp.asarray(acts[t]), general_trade=True)
        states.append(state); infos.append(info)
        if end is None and int(info.winner) >= 0: end = (t, int(info.winner))
    if end is not None:
        w = end[1]; who = names[w] if w < N else f"team {w}"
        print(f"ends on turn {end[0]}: {who} wins" + (" (team)" if row.get("teams") else ""))
    else:
        leavers = [names[a["index"]] for a in row["afks"] if a["index"] < N]
        print("no capture in the record" + (f"; left the game: {', '.join(dict.fromkeys(leavers))}" if leavers else ""))
    if a.no_gui: return

    from generals.gui import ReplayGUI
    from generals.gui.properties import GuiMode
    labels = list(names) + [f"(pad {i})" for i in range(N, n_pad)]
    gui = ReplayGUI(states[0], agent_ids=labels, fps=a.fps, mode=GuiMode.REPLAY, start_paused=a.paused)
    gui.play(states, infos)      # SPACE play/pause, arrows step, R restart, Q quit


if __name__ == "__main__":
    main()
