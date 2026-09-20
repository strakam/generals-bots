#!/usr/bin/env python
"""Replay current-format generals.io games (.gior, format 16+) through the engine under today's rules.

For every game the recorded moves are fed through game.step in order. A recorded move is counted as
*skipped* when, at the start of its tick, the mover does not own its source with at least 2 armies (the
site drops such queued moves as well, so a skip is only a divergence if it cascades). Agreement per game:
  capture games   the simulated game ends by a capture by the recorded winner on the recorded turn;
  surrender games the record carries the site's surrender/afk event and no capture; agreement means the
                  simulation has not ended before the recorded end and no move was skipped.
Usage: python replay_recent.py 'dir/*.gior' [--no-trade] [--legacy] [--official] [--out f]
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import argparse, glob, json, sys  # noqa: E402
import numpy as np  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gior  # noqa: E402
import replay_agreement as ra  # noqa: E402
ra.PAD = 32   # recent maps exceed the archive's 23x23 padding
import jax.numpy as jnp  # noqa: E402
from generals.core import game  # noqa: E402


import jax  # noqa: E402
from functools import partial  # noqa: E402

BUCKET = 512   # actions are padded to a multiple of this many ticks so the scan compiles once per bucket


@partial(jax.jit, static_argnames=("trade", "legacy", "official"))
def simulate(grid, actions, trade, legacy, official, tunnel_limits=None):
    """Whole game as one lax.scan. Returns per tick: legality of each recorded move at the start of
    the tick (mover owns the source with >= 2 armies), the winner after the tick, and the general
    positions at the start of the tick."""
    s0 = game.create_initial_state(grid, tunnel_limits=tunnel_limits)

    def tick(s, a):
        si, sj = a[:, 1], a[:, 2]
        legal = (a[:, 0] != 0) | (s.ownership[jnp.arange(2), si, sj] & (s.armies[si, sj] >= 2))
        s2, info = game.step(s, a, legacy_move_priority=legacy, general_trade=trade, official_move_priority=official)
        return s2, (legal, info.winner, s.general_positions)

    _, (legal, winner, gp) = jax.lax.scan(tick, s0, actions)
    return legal, winner, gp


def run(path, trade=True, legacy=False, official=False):
    row = gior.row(path); prep = ra.prepare(row)
    if prep["exclude"]: return dict(id=row["id"], excluded=prep["exclude"])
    grid, actions, meta = prep["grid"], prep["actions"], prep["meta"]; W = row["mapWidth"]
    T = actions.shape[0]; Tp = -(-T // BUCKET) * BUCKET
    pad = np.zeros((Tp - T, 2, 5), dtype=actions.dtype); pad[:, :, 0] = 1          # pass actions
    acts = np.concatenate([actions, pad], axis=0)
    tl = prep["tunnel_limits"]; tl = None if tl is None else jnp.asarray(tl)
    legal, winner, gp = simulate(jnp.asarray(grid), jnp.asarray(acts), trade, legacy, official, tl)
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
    ended_by_event = any(a["turn"] > int(last[4]) for a in row["afks"])          # an afk on the capture turn itself is the loser leaving
    capture = int(last[2]) == opp_gen and not ended_by_event; rec_winner = mover if capture else -1
    surrender = not capture
    if surrender: agree = end[0] < 0 and not skipped
    else: agree = end == (int(last[4]), rec_winner)
    return dict(id=row["id"], version=row["version"], n_moves=meta["n_moves"], surrender=surrender, afks=row["afks"], extras=row["extras"],
                recorded_end=int(last[4]), recorded_winner=rec_winner, sim_end=end[0], sim_winner=end[1],
                skipped=skipped, agree=bool(agree))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("pattern"); ap.add_argument("--no-trade", action="store_true"); ap.add_argument("--legacy", action="store_true"); ap.add_argument("--official", action="store_true"); ap.add_argument("--out")
    a = ap.parse_args(); res = [run(f, trade=not a.no_trade, legacy=a.legacy, official=a.official) for f in sorted(glob.glob(a.pattern))]
    ok = [r for r in res if "excluded" not in r]; cap = [r for r in ok if not r["surrender"]]; sur = [r for r in ok if r["surrender"]]
    print(f"games {len(ok)} (excluded {len(res)-len(ok)}); moves {sum(r['n_moves'] for r in ok)}; skipped moves {sum(len(r['skipped']) for r in ok)}")
    print(f"capture games {len(cap)}: same winner on the same turn {sum(r['agree'] for r in cap)}")
    print(f"surrender games {len(sur)}: no early end and no skipped move {sum(r['agree'] for r in sur)}; sim ended early {sum(1 for r in sur if r['sim_end'] >= 0)}")
    for r in ok:
        if not r["agree"]: print("  DISAGREE", r["id"], "surrender" if r["surrender"] else "capture", "rec", r["recorded_end"], r["recorded_winner"], "sim", r["sim_end"], r["sim_winner"], "skipped", r["skipped"][:4])
    if a.out: json.dump(res, open(a.out, "w"), indent=1)
