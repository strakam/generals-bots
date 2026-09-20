#!/usr/bin/env python
"""Compare our engine with generals.io's own replay engine tile by tile, tick by tick.

The site's boards come from running the client bundle's Game/Map/MoveResolver under Node
(scratch tool run_many.js): one JSON per game with owners[t] and armies[t] for t = 0..turns,
where entry t is the board after the moves recorded with turn t-1, i.e. our state at the start
of tick t. Usage: python board_diff.py <siteboards dir> 'gior/*.gior' [--out f]
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import argparse, glob, json, sys  # noqa: E402
import numpy as np  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gior  # noqa: E402
import replay_recent as rr  # noqa: E402
import jax, jax.numpy as jnp  # noqa: E402
from functools import partial  # noqa: E402
from generals.core import game  # noqa: E402


@partial(jax.jit, static_argnames=("trade", "legacy"))
def boards(grid, actions, trade, legacy):
    s0 = game.create_initial_state(grid)

    def tick(s, a):
        s2, _ = game.step(s, a, legacy_move_priority=legacy, general_trade=trade)
        return s2, (s.ownership, s.armies)

    _, (own, arm) = jax.lax.scan(tick, s0, actions)
    return own, arm


def run(path, site_dir):
    row = gior.row(path); prep = rr.ra.prepare(row)
    if prep["exclude"]: return dict(id=row["id"], excluded=prep["exclude"])
    sp = os.path.join(site_dir, row["id"] + ".json")
    if not os.path.exists(sp): return dict(id=row["id"], excluded="no site board")
    site = json.load(open(sp)); W, H = site["W"], site["H"]
    so, sa = np.array(site["owners"]), np.array(site["armies"])                     # (turns+1, H*W)
    T = min(so.shape[0], prep["actions"].shape[0] + 1)
    acts = np.zeros((-(-T // rr.BUCKET) * rr.BUCKET, 2, 5), dtype=np.int32); acts[:, :, 0] = 1
    acts[:prep["actions"].shape[0]] = prep["actions"]
    own, arm = boards(jnp.asarray(prep["grid"]), jnp.asarray(acts), True, False)
    own, arm = np.asarray(own)[:T, :, :H, :W].reshape(T, 2, -1), np.asarray(arm)[:T, :H, :W].reshape(T, -1)
    oo = np.where(own[:, 0], 0, np.where(own[:, 1], 1, -1))
    so_ = np.where(so[:T] < 0, -1, so[:T])                                            # site: -1 empty, -2 mountain, -3/-4 fog codes
    bad = (oo != so_) | (arm != sa[:T])
    # The site's terminal board (after the capture that ends the game) still receives that turn's
    # general increment; our engine stops at the capture. Compare all in-game ticks, and the
    # terminal tick separately.
    in_game = bad[:-1] if site["winner"] else bad
    first = int(np.argmax(in_game.any(1))) if in_game.any() else -1
    return dict(id=row["id"], turns=int(T - 1), site_turns=int(site["turns"]), site_winner=site["winner"], consumed=site["consumed"],
                n_moves=site["n_moves"], mismatch_ticks=int(in_game.any(1).sum()), first_mismatch=first,
                first_tiles=[int(i) for i in np.nonzero(bad[first])[0][:5]] if first >= 0 else [],
                terminal_mismatch=bool(bad[-1].any()) if site["winner"] else False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("site_dir"); ap.add_argument("pattern")
    ap.add_argument("--out")
    a = ap.parse_args(); res = [run(f, a.site_dir) for f in sorted(glob.glob(a.pattern))]
    ok = [r for r in res if "excluded" not in r]
    ident = [r for r in ok if r["mismatch_ticks"] == 0]
    print(f"games {len(ok)} (excluded {len(res)-len(ok)}); ticks {sum(r['turns'] for r in ok)}; identical boards at every in-game tick: {len(ident)}; terminal board differs (post-capture increment): {sum(r['terminal_mismatch'] for r in ok)}")
    for r in ok:
        if r["mismatch_ticks"]: print("  DIFF", r["id"], "first mismatch tick", r["first_mismatch"], "tiles", r["first_tiles"], "mismatching ticks", r["mismatch_ticks"], "/", r["turns"])
    if a.out: json.dump(res, open(a.out, "w"))
