#!/usr/bin/env python
"""Replay current-format generals.io games (.gior, format 16+) through the engine under today's rules.

For every game the recorded moves are fed through game.step in order. A recorded move is counted as
*skipped* when, at the start of its tick, the mover does not own its source with at least 2 armies (the
site drops such queued moves as well, so a skip is only a divergence if it cascades). Agreement per game:
  capture games   the simulated game ends by a capture by the recorded winner on the recorded turn;
  surrender games the record carries the site's surrender/afk event and no capture; agreement means the
                  simulation has not ended before the recorded end and no move was skipped.
Usage: python replay_recent.py 'dir/*.gior' [--no-trade] [--out f]
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import argparse, glob, json, sys  # noqa: E402
import numpy as np  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gior  # noqa: E402
import replay_prep as ra  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from generals.core import game  # noqa: E402


import jax  # noqa: E402
from functools import partial  # noqa: E402

BUCKET = 512   # actions are padded to a multiple of this many ticks so the scan compiles once per bucket


@partial(jax.jit, static_argnames=("trade",))
def simulate(grid, actions, trade):
    """Whole game as one lax.scan. Returns per tick: legality of each recorded move at the start of
    the tick (mover owns the source with >= 2 armies), the winner after the tick, and the general
    positions at the start of the tick."""
    s0 = game.create_initial_state(grid)

    def tick(s, a):
        si, sj = a[:, 1], a[:, 2]
        legal = (a[:, 0] != 0) | (s.ownership[jnp.arange(2), si, sj] & (s.armies[si, sj] >= 2))
        s2, info = game.step(s, a, general_trade=trade)
        return s2, (legal, info.winner, s.general_positions)

    _, (legal, winner, gp) = jax.lax.scan(tick, s0, actions)
    return legal, winner, gp


def run(path, trade=True):
    row = gior.row(path); prep = ra.prepare(row)
    if prep["exclude"]: return dict(id=row["id"], excluded=prep["exclude"])
    grid, actions, meta = prep["grid"], prep["actions"], prep["meta"]; W = row["mapWidth"]
    T = actions.shape[0]; Tp = -(-T // BUCKET) * BUCKET
    pad = np.zeros((Tp - T, 2, 5), dtype=actions.dtype); pad[:, :, 0] = 1          # pass actions
    acts = np.concatenate([actions, pad], axis=0)
    legal, winner, gp = simulate(jnp.asarray(grid), jnp.asarray(acts), trade)
    legal, winner, gp = np.asarray(legal), np.asarray(winner), np.asarray(gp)
    ends = np.nonzero(winner >= 0)[0]
    end = (int(ends[0]), int(winner[ends[0]])) if len(ends) else (-1, -1)
    last = row["moves"][-1]; gp_last = gp[int(last[4])]                             # generals as they stand when the last move resolves
    horizon = T if end[0] < 0 else min(T, end[0] + 1)                               # skips are counted up to and including the end tick
    skipped = [(int(t), int(p)) for t, p in zip(*np.nonzero(~legal[:horizon]))]
    # A capture is a last move onto the OTHER player's general as it stands at that tick (after a
    # general trade the generals have swapped); the mover is the recorded winner. A surrender/afk
    # event after the last move means the game ended by that event, whatever the last move was.
    mover = int(last[0]); opp_gen = int(gp_last[1 - mover][0]) * W + int(gp_last[1 - mover][1])
    t_last = int(last[4]); hits_general = int(last[2]) == opp_gen
    afk_after = any(a["turn"] > t_last for a in row["afks"])
    afk_same = any(a["turn"] == t_last for a in row["afks"])
    # Three endings can be read off the record: a surrender/afk event after the last move; a last
    # move onto the opponent's general with no event (a capture); and a last move onto the general
    # with an afk on that same turn, which the record cannot tell apart (the loser leaves as the
    # general falls, or resigns as a final assault fails) -- the site's own engine settles those in
    # board_diff.py.
    if afk_after or not hits_general:
        kind, agree = "surrender", end[0] < 0 and not skipped
    elif afk_same:
        kind, agree = "capture_or_surrender", (end == (-1, -1) or end == (t_last, mover)) and not skipped
    else:
        kind, agree = "capture", end == (t_last, mover)
    capture = kind == "capture"; rec_winner = mover if capture else -1; surrender = kind == "surrender"
    return dict(id=row["id"], version=row["version"], n_moves=meta["n_moves"], surrender=surrender, kind=kind, afks=row["afks"], extras=row["extras"],
                recorded_end=int(last[4]), recorded_winner=rec_winner, sim_end=end[0], sim_winner=end[1],
                skipped=skipped, agree=bool(agree))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("pattern"); ap.add_argument("--no-trade", action="store_true"); ap.add_argument("--out")
    a = ap.parse_args(); res = [run(f, trade=not a.no_trade) for f in sorted(glob.glob(a.pattern))]
    ok = [r for r in res if "excluded" not in r]
    cap = [r for r in ok if r["kind"] == "capture"]; sur = [r for r in ok if r["kind"] == "surrender"]; amb = [r for r in ok if r["kind"] == "capture_or_surrender"]
    print(f"games {len(ok)} (excluded {len(res)-len(ok)}); moves {sum(r['n_moves'] for r in ok)}; skipped moves {sum(len(r['skipped']) for r in ok)}")
    print(f"capture games {len(cap)}: same winner on the same turn {sum(r['agree'] for r in cap)}")
    print(f"surrender games {len(sur)}: no early end and no skipped move {sum(r['agree'] for r in sur)}; sim ended early {sum(1 for r in sur if r['sim_end'] >= 0)}")
    print(f"last move onto the general with an afk that turn {len(amb)}: no early end, ends (if at all) by that move {sum(r['agree'] for r in amb)}")
    for r in ok:
        if not r["agree"]: print("  DISAGREE", r["id"], r["kind"], "rec", r["recorded_end"], r["recorded_winner"], "sim", r["sim_end"], r["sim_winner"], "skipped", r["skipped"][:4])
    if a.out: json.dump(res, open(a.out, "w"), indent=1)
