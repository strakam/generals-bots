#!/usr/bin/env python
"""Replay current-format generals.io games (.gior, format 16+) through the engine under today's rules.

For every game the recorded moves are fed through game.step in order. A recorded move is counted as
*skipped* when, at the start of its tick, the mover does not own its source with at least 2 armies (the
site drops such queued moves as well, so a skip is only a divergence if it cascades). Agreement per game:
  capture games   the simulated game ends by a capture by the recorded winner on the recorded turn;
  surrender games the record carries the site's surrender/afk event and no capture; agreement means the
                  simulation has not ended before the recorded end and no move was skipped.
Usage: python replay_recent.py 'dir/*.gior' [--no-trade] [--legacy]
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import argparse, glob, json, sys  # noqa: E402
import numpy as np  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gior  # noqa: E402
import replay_agreement as ra  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from generals.core import game  # noqa: E402


def run(path, trade=True, legacy=False):
    row = gior.row(path); prep = ra.prepare(row)
    if prep["exclude"]: return dict(id=row["id"], excluded=prep["exclude"])
    grid, actions, meta = prep["grid"], prep["actions"], prep["meta"]; W = row["mapWidth"]
    s = game.create_initial_state(jnp.asarray(grid)); skipped = []; end = (-1, -1)
    last = row["moves"][-1]; gp_last = None
    for t in range(actions.shape[0]):
        arm, own = np.asarray(s.armies), np.asarray(s.ownership)
        if t == int(last[4]): gp_last = np.asarray(s.general_positions)     # generals as they stand when the last move resolves
        for p in (0, 1):
            a = actions[t, p]
            if a[0] == 0 and end[0] < 0 and not (own[p, a[1], a[2]] and arm[a[1], a[2]] >= 2): skipped.append((t, p))
        s, info = game.step(s, jnp.asarray(actions[t]), legacy_move_priority=legacy, general_trade=trade)
        if end[0] < 0 and int(info.winner) >= 0: end = (t, int(info.winner))
    # A capture is a last move onto the OTHER player's general as it stands at that tick (after a
    # general trade the generals have swapped); the mover is the recorded winner.
    mover = int(last[0]); opp_gen = int(gp_last[1 - mover][0]) * W + int(gp_last[1 - mover][1])
    capture = int(last[2]) == opp_gen; rec_winner = mover if capture else -1
    surrender = not capture
    if surrender: agree = end[0] < 0 and not skipped
    else: agree = end == (int(last[4]), rec_winner)
    return dict(id=row["id"], version=row["version"], n_moves=meta["n_moves"], surrender=surrender, afks=row["afks"], extras=row["extras"],
                recorded_end=int(last[4]), recorded_winner=rec_winner, sim_end=end[0], sim_winner=end[1],
                skipped=skipped, agree=bool(agree))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("pattern"); ap.add_argument("--no-trade", action="store_true"); ap.add_argument("--legacy", action="store_true"); ap.add_argument("--out")
    a = ap.parse_args(); res = [run(f, trade=not a.no_trade, legacy=a.legacy) for f in sorted(glob.glob(a.pattern))]
    ok = [r for r in res if "excluded" not in r]; cap = [r for r in ok if not r["surrender"]]; sur = [r for r in ok if r["surrender"]]
    print(f"games {len(ok)} (excluded {len(res)-len(ok)}); moves {sum(r['n_moves'] for r in ok)}; skipped moves {sum(len(r['skipped']) for r in ok)}")
    print(f"capture games {len(cap)}: same winner on the same turn {sum(r['agree'] for r in cap)}")
    print(f"surrender games {len(sur)}: no early end and no skipped move {sum(r['agree'] for r in sur)}; sim ended early {sum(1 for r in sur if r['sim_end'] >= 0)}")
    for r in ok:
        if not r["agree"]: print("  DISAGREE", r["id"], "surrender" if r["surrender"] else "capture", "rec", r["recorded_end"], r["recorded_winner"], "sim", r["sim_end"], r["sim_winner"], "skipped", r["skipped"][:4])
    if a.out: json.dump(res, open(a.out, "w"), indent=1)
