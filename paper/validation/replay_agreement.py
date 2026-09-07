#!/usr/bin/env python
"""Replay agreement: how faithfully does the JAX engine reproduce archived generals.io 1v1 games?

Every replay in the `strakammm/generals_io_replays` dataset is turned into a
starting grid plus one action per (tick, player) and fed through the engine,
move for move. Nothing in the recorded state is available beyond the moves, so
faithfulness is judged by two things a wrong engine cannot fake:

  * legality — is each recorded move still playable in the simulated board at
    the moment it resolves (the mover owns the source cell with >= 2 armies and
    the destination is on the board and not a mountain)? A human only ever
    sends moves that are legal on the board they see, so the first illegal
    recorded move marks the first tick at which the simulated board no longer
    matches the one the players saw.
  * the ending — every replay in the archive ends with a general capture. Does
    the engine declare the same winner on the same tick?

Two move-resolution rules are simulated side by side from the same actions:

  current  (generals/core/game.py::_determine_move_order, default)
           chasing > reinforcing > smaller army, ties by player index.
  legacy   (the same function with legacy_move_priority=True)
           priority alternates each tick: P0 resolves first on even ticks,
           P1 on odd ticks. This is the rule the archive was recorded under
           and the engine used before generals-bots commit e5676c3 (2025-04).

Because the legacy run reproduces the archive exactly (see README), the first
tick on which the two simulated boards differ is the exact first divergence of
the current rule from the recorded game, and by construction it is a tick on
which the two players' moves interacted.

Conventions (established on 5 replays, see README "Alignment"):
  * a move recorded with turn t is played in the engine step taken from
    state.time == t, i.e. the (t+1)-th call of step(). generals.io's tick
    counter and state.time agree; the first production lands on tick 2 in
    both, so the earliest recorded moves (turn 2) are legal.
  * grids are padded to PAD x PAD with mountains (the engine JIT-compiles per
    shape; mountains are inert) and the tick axis to a multiple of T_BUCKET
    with pass actions.
  * a move by a player who is eliminated at the moment it resolves (their
    general fell earlier in the same tick) is not counted as illegal, and
    moves recorded after the simulated game has already ended are counted
    separately (`*_n_moves_after_end`) rather than as illegal.
  * if the archive ever held two moves by one player on one tick the first
    would be kept and the rest counted in `n_dropped_duplicate`; the dataset
    contains none.

Usage:
    python replay_agreement.py --out paper/validation/results            # full run
    python replay_agreement.py --limit 200 --workers 4 --out /tmp/x       # smoke test
    python replay_agreement.py --selftest 5                               # alignment check

Outputs `per_game.csv` (one row per replay) and `summary.json`.
"""
from __future__ import annotations

import os

# JAX must see these before it is imported, in the parent and (spawned) workers.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse  # noqa: E402
import csv  # noqa: E402
import glob  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import multiprocessing as mp  # noqa: E402
import sys  # noqa: E402
import time as _time  # noqa: E402
from collections import Counter  # noqa: E402
from functools import partial  # noqa: E402

import numpy as np  # noqa: E402

PARQUET_GLOB = os.path.expanduser(
    "~/.cache/huggingface/hub/datasets--strakammm--generals_io_replays/snapshots/*/data/train-00000-of-00001.parquet"
)
PAD = 23          # largest map side in the dataset; every grid is padded to PAD x PAD with mountains
T_BUCKET = 100    # tick axis padded to a multiple of this (pass actions) to bound JIT compilations
MOUNTAIN, EMPTY = -2, 0
# (drow, dcol) -> engine direction index (generals.core.action.DIRECTIONS = UP, DOWN, LEFT, RIGHT)
DELTA_TO_DIR = {(-1, 0): 0, (1, 0): 1, (0, -1): 2, (0, 1): 3}
PASS = (1, 0, 0, 0, 0)
# legality reason codes emitted by the scan
LEGAL, NOT_OWNED, TOO_FEW, BLOCKED = 0, 1, 2, 3
REASON_NAME = {LEGAL: "legal", NOT_OWNED: "source not owned", TOO_FEW: "source has <2 armies",
               BLOCKED: "destination off-board or mountain"}


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def parquet_path() -> str:
    paths = glob.glob(PARQUET_GLOB)
    if not paths:
        raise FileNotFoundError(f"no parquet matches {PARQUET_GLOB}")
    return paths[0]


def load_rows(path: str, lo: int = 0, hi: int | None = None) -> list[dict]:
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    hi = table.num_rows if hi is None else min(hi, table.num_rows)
    return table.slice(lo, hi - lo).to_pylist()


def prepare(row: dict) -> dict:
    """Turn one replay row into engine inputs, or record why it is excluded.

    Returns a dict with `exclude` (None or a reason string) and, when usable,
    `grid` (PAD x PAD int32), `actions` (T x 2 x 5 int32, T = last turn + 1),
    `meta` (per-game bookkeeping).
    """
    w, h = int(row["mapWidth"]), int(row["mapHeight"])
    usernames = row["usernames"] or []
    generals = list(row["generals"] or [])
    meta = {"id": row["id"], "version": int(row["version"]), "map_w": w, "map_h": h,
            "n_players": len(usernames)}
    if len(usernames) != 2:
        return {"exclude": f"{len(usernames)} players", "meta": meta}
    if len(generals) != 2 or any(g < 0 or g >= w * h for g in generals):
        return {"exclude": "generals not two on-board tiles", "meta": meta}
    if w > PAD or h > PAD:
        return {"exclude": f"map larger than {PAD}x{PAD}", "meta": meta}

    cities, city_armies = list(row["cities"] or []), list(row["cityArmies"] or [])
    mountains = list(row["mountains"] or [])
    if len(cities) != len(city_armies):
        return {"exclude": "cities/cityArmies length mismatch", "meta": meta}
    if any(a <= 2 for a in city_armies):
        # the engine reads grid values > N(=2) as castles; a smaller garrison could not be encoded
        return {"exclude": "city army <= 2 (not encodable)", "meta": meta}
    special = set(cities) | set(mountains) | set(generals)
    if len(special) != len(cities) + len(mountains) + len(generals):
        return {"exclude": "overlapping city/mountain/general tiles", "meta": meta}
    if any(t < 0 or t >= w * h for t in special):
        return {"exclude": "tile index off the map", "meta": meta}

    grid = np.full((PAD, PAD), MOUNTAIN, dtype=np.int32)
    board = np.full(h * w, EMPTY, dtype=np.int32)
    board[mountains] = MOUNTAIN
    for t, a in zip(cities, city_armies):
        board[t] = int(a)
    board[generals[0]] = 1
    board[generals[1]] = 2
    grid[:h, :w] = board.reshape(h, w)

    moves = row["moves"] or []
    if not moves:
        return {"exclude": "no moves", "meta": meta}
    turns = [m[4] for m in moves]
    if turns != sorted(turns):
        return {"exclude": "moves not sorted by turn", "meta": meta}

    T = int(turns[-1]) + 1
    actions = np.tile(np.array(PASS, dtype=np.int32), (T, 2, 1))
    seen = set()
    n_dup = 0
    n_split = 0
    per_player = [0, 0]
    for p, s, e, is50, t in moves:
        p, s, e, t = int(p), int(s), int(e), int(t)
        if p not in (0, 1):
            return {"exclude": f"move by player index {p}", "meta": meta}
        if t < 0:
            return {"exclude": "negative turn", "meta": meta}
        if not (0 <= s < w * h and 0 <= e < w * h):
            return {"exclude": "move tile off the map", "meta": meta}
        r0, c0 = divmod(s, w)
        r1, c1 = divmod(e, w)
        d = DELTA_TO_DIR.get((r1 - r0, c1 - c0))
        if d is None:
            return {"exclude": "non-adjacent move", "meta": meta}
        if (p, t) in seen:
            n_dup += 1            # policy: first move of a (player, tick) pair wins
            continue
        seen.add((p, t))
        actions[t, p] = (0, r0, c0, d, 1 if is50 else 0)
        n_split += 1 if is50 else 0
        per_player[p] += 1

    last = moves[-1]
    lp = int(last[0])
    last_hits_general = int(last[2]) == generals[1 - lp]
    meta.update(
        n_moves=len(moves) - n_dup, n_dropped_duplicate=n_dup, n_moves_p0=per_player[0],
        n_moves_p1=per_player[1], n_split_moves=n_split, n_ticks=T,
        recorded_end_turn=int(last[4]), recorded_winner=lp if last_hits_general else -1,
        recorded_end_is_capture=bool(last_hits_general),
    )
    return {"exclude": None, "grid": grid, "actions": actions, "meta": meta}


def interaction_flags(actions: np.ndarray) -> dict[str, np.ndarray]:
    """Per tick, from the recorded moves alone: did the two moves touch a common cell?"""
    a0, a1 = actions[:, 0], actions[:, 1]
    moved = (a0[:, 0] == 0) & (a1[:, 0] == 0)
    dr = np.array([-1, 1, 0, 0])
    dc = np.array([0, 0, -1, 1])
    s0 = a0[:, 1] * PAD + a0[:, 2]
    s1 = a1[:, 1] * PAD + a1[:, 2]
    d0 = (a0[:, 1] + dr[a0[:, 3]]) * PAD + (a0[:, 2] + dc[a0[:, 3]])
    d1 = (a1[:, 1] + dr[a1[:, 3]]) * PAD + (a1[:, 2] + dc[a1[:, 3]])
    same_dest = moved & (d0 == d1)
    chase0 = moved & (d0 == s1)          # P0 moves onto the cell P1 is leaving
    chase1 = moved & (d1 == s0)
    same_src = moved & (s0 == s1)        # both command the same cell (only one can own it)
    shared = same_dest | chase0 | chase1 | same_src
    return {"both_moved": moved, "same_dest": same_dest, "chase0": chase0, "chase1": chase1,
            "same_src": same_src, "shared_cell": shared}


# --------------------------------------------------------------------------- #
# Simulation (one jitted scan per tick-bucket)
# --------------------------------------------------------------------------- #
_RUNNERS: dict[int, object] = {}


def _make_runner(T: int):
    import jax
    import jax.numpy as jnp
    from jax import lax

    from generals.core import game
    from generals.core.action import compute_valid_move_mask

    def resolve(state, acts, legacy):
        """Mirror game.step's resolution loop, probing legality as each move is played."""
        order = game._determine_move_order(state, acts, legacy)
        reason = jnp.zeros(2, jnp.int32)
        elim = jnp.zeros(2, bool)
        for k in range(2):
            p = order[k]
            a = acts[p]
            si = jnp.clip(a[1], 0, PAD - 1)
            sj = jnp.clip(a[2], 0, PAD - 1)
            mask = compute_valid_move_mask(state.armies, state.ownership[p], state.mountains)
            owned = state.ownership[p, si, sj]
            enough = state.armies[si, sj] > 1
            ok = mask[si, sj, a[3]]
            code = jnp.where(ok, LEGAL, jnp.where(~owned, NOT_OWNED, jnp.where(~enough, TOO_FEW, BLOCKED)))
            reason = reason.at[p].set(jnp.where(a[0] == 0, code, LEGAL))
            elim = elim.at[p].set(state.eliminated[p])
            state = game.execute_action(state, p, a)
        return state, reason, elim, order[0]

    def probe(state, acts):
        """Pre-tick facts about each player's move on the recorded (legacy) board:
        source army, whether the source is a general, destination owner (-1 neutral),
        destination army, whether the destination is a general."""
        src_army = jnp.zeros(2, jnp.int32)
        src_gen = jnp.zeros(2, bool)
        dst_owner = jnp.zeros(2, jnp.int32)
        dst_army = jnp.zeros(2, jnp.int32)
        dst_gen = jnp.zeros(2, bool)
        dst_castle = jnp.zeros(2, bool)
        for p in range(2):
            a = acts[p]
            si = jnp.clip(a[1], 0, PAD - 1)
            sj = jnp.clip(a[2], 0, PAD - 1)
            di = jnp.clip(si + game.DIRECTIONS[a[3], 0], 0, PAD - 1)
            dj = jnp.clip(sj + game.DIRECTIONS[a[3], 1], 0, PAD - 1)
            owner = jnp.where(state.ownership[0, di, dj], 0, jnp.where(state.ownership[1, di, dj], 1, -1))
            src_army = src_army.at[p].set(state.armies[si, sj])
            src_gen = src_gen.at[p].set(state.generals[si, sj])
            dst_owner = dst_owner.at[p].set(owner)
            dst_army = dst_army.at[p].set(state.armies[di, dj])
            dst_gen = dst_gen.at[p].set(state.generals[di, dj])
            dst_castle = dst_castle.at[p].set(state.castles[di, dj])
        return dict(src_army=src_army, src_is_general=src_gen, dst_owner=dst_owner, dst_army=dst_army,
                    dst_is_general=dst_gen, dst_is_castle=dst_castle)

    def prestep_reason(state, acts):
        out = jnp.zeros(2, jnp.int32)
        for p in range(2):
            a = acts[p]
            si = jnp.clip(a[1], 0, PAD - 1)
            sj = jnp.clip(a[2], 0, PAD - 1)
            mask = compute_valid_move_mask(state.armies, state.ownership[p], state.mountains)
            owned = state.ownership[p, si, sj]
            enough = state.armies[si, sj] > 1
            ok = mask[si, sj, a[3]]
            code = jnp.where(ok, LEGAL, jnp.where(~owned, NOT_OWNED, jnp.where(~enough, TOO_FEW, BLOCKED)))
            out = out.at[p].set(jnp.where(a[0] == 0, code, LEGAL))
        return out

    def finish(state):
        """The tail of game.step: tick counter and production."""
        done_before = state.winner >= 0
        state = lax.cond(done_before, lambda s: s, lambda s: s._replace(time=s.time + 1), state)
        return lax.cond(state.winner >= 0, lambda s: s, game.global_update, state)

    def one_tick(carry, acts):
        cur, leg, ever = carry           # ever: (2, PAD, PAD) cells each player has held so far (legacy board)
        ever = ever | leg.ownership
        src_ever = jnp.stack([ever[p, jnp.clip(acts[p, 1], 0, PAD - 1), jnp.clip(acts[p, 2], 0, PAD - 1)]
                              for p in range(2)])
        pre_cur = prestep_reason(cur, acts)
        pre_leg = prestep_reason(leg, acts)
        done_cur = cur.winner >= 0
        done_leg = leg.winner >= 0
        leg_probe = probe(leg, acts)
        cur, r_cur, e_cur, first_cur = resolve(cur, acts, False)
        leg, r_leg, e_leg, first_leg = resolve(leg, acts, True)
        cur = finish(cur)
        leg = finish(leg)
        # Boards differ. Once BOTH games are over the boards are allowed to differ
        # (the spoils are halved in whichever order the last tick resolved) — only
        # a difference while at least one game is still running is a divergence.
        boards_differ = jnp.any(cur.armies != leg.armies) | jnp.any(cur.ownership != leg.ownership)
        diverged = boards_differ & ~((cur.winner >= 0) & (leg.winner >= 0))
        out = dict(
            cur_reason=r_cur, cur_elim=e_cur, cur_pre=pre_cur, cur_done_before=done_cur, cur_winner=cur.winner,
            leg_reason=r_leg, leg_elim=e_leg, leg_pre=pre_leg, leg_done_before=done_leg, leg_winner=leg.winner,
            cur_first_mover=first_cur, leg_first_mover=first_leg,
            diverged=diverged,
            cur_army=jnp.sum(cur.armies[None] * cur.ownership, axis=(1, 2)),
            **{"leg_" + k: v for k, v in leg_probe.items()},
        )
        out["leg_src_ever_owned"] = src_ever
        return (cur, leg, ever), out

    @jax.jit
    def run(grid, actions):
        s0 = game.create_initial_state(grid)
        ever0 = jnp.zeros((2, PAD, PAD), dtype=bool)
        _, out = lax.scan(one_tick, (s0, s0, ever0), actions)
        return out

    return run


def simulate(grid: np.ndarray, actions: np.ndarray) -> dict[str, np.ndarray]:
    T = actions.shape[0]
    Tp = int(math.ceil(T / T_BUCKET) * T_BUCKET)
    if Tp not in _RUNNERS:
        _RUNNERS[Tp] = _make_runner(Tp)
    padded = np.tile(np.array(PASS, dtype=np.int32), (Tp, 2, 1))
    padded[:T] = actions
    out = _RUNNERS[Tp](grid, padded)
    return {k: np.asarray(v)[:T] for k, v in out.items()}


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def _rule_metrics(prefix: str, out: dict, actions: np.ndarray, meta: dict) -> dict:
    reason = out[f"{prefix}_reason"]                  # (T, 2)
    elim = out[f"{prefix}_elim"]
    pre = out[f"{prefix}_pre"]
    done_before = out[f"{prefix}_done_before"]        # (T,)
    winner = out[f"{prefix}_winner"]
    moved = actions[:, :, 0] == 0                     # (T, 2)
    T = actions.shape[0]

    ended = np.nonzero(winner >= 0)[0]
    sim_end_turn = int(ended[0]) if len(ended) else -1
    sim_winner = int(winner[ended[0]]) if len(ended) else -1
    after_end = moved & done_before[:, None]
    # illegal: a real move, judged illegal as it resolved, by a player who was
    # not already eliminated at that moment, before the simulated game ended
    illegal = moved & (reason != LEGAL) & ~elim & ~done_before[:, None]
    illegal_pre = moved & (pre != LEGAL) & ~done_before[:, None]
    ill_idx = np.argwhere(illegal)
    first_ill = int(ill_idx[0, 0]) if len(ill_idx) else -1
    first_ill_player = int(ill_idx[0, 1]) if len(ill_idx) else -1
    first_ill_reason = REASON_NAME[int(reason[ill_idx[0, 0], ill_idx[0, 1]])] if len(ill_idx) else ""

    rec_end = meta["recorded_end_turn"]
    m = {
        f"{prefix}_n_illegal": int(illegal.sum()),
        f"{prefix}_first_illegal_turn": first_ill,
        f"{prefix}_first_illegal_player": first_ill_player,
        f"{prefix}_first_illegal_reason": first_ill_reason,
        f"{prefix}_n_illegal_not_owned": int((illegal & (reason == NOT_OWNED)).sum()),
        f"{prefix}_n_illegal_too_few": int((illegal & (reason == TOO_FEW)).sum()),
        f"{prefix}_n_illegal_blocked": int((illegal & (reason == BLOCKED)).sum()),
        # a "not owned" source the mover never held on the (legacy) board: the log must be missing
        # the move that took it, or the real game resolved an earlier clash differently
        f"{prefix}_n_illegal_source_never_held": int((illegal & (reason == NOT_OWNED) & ~out["leg_src_ever_owned"]).sum()),
        f"{prefix}_n_illegal_prestep": int(illegal_pre.sum()),
        f"{prefix}_n_moves_after_end": int(after_end.sum()),
        f"{prefix}_sim_end_turn": sim_end_turn,
        f"{prefix}_sim_winner": sim_winner,
        f"{prefix}_never_ended": sim_end_turn < 0,
        f"{prefix}_ended_exact": sim_end_turn == rec_end,
        f"{prefix}_ended_same_turn": sim_end_turn >= 0 and abs(sim_end_turn - rec_end) <= 1,
        f"{prefix}_ended_early": 0 <= sim_end_turn < rec_end,
        f"{prefix}_ended_late": sim_end_turn > rec_end,
        f"{prefix}_winner_agrees": sim_end_turn >= 0 and sim_winner == meta["recorded_winner"],
    }
    m[f"{prefix}_agrees"] = (m[f"{prefix}_n_illegal"] == 0 and m[f"{prefix}_ended_exact"]
                             and m[f"{prefix}_winner_agrees"])
    return m


def _divergence_kind(flags: dict, t: int) -> str:
    if t < 0:
        return "none"
    if not flags["both_moved"][t]:
        return "only one player moved"
    if flags["same_dest"][t]:
        return "both moved into the same cell"
    if flags["chase0"][t] and flags["chase1"][t]:
        return "head-on: each moved into the other's source"
    if flags["chase0"][t] or flags["chase1"][t]:
        return "one moved into the cell the other was leaving"
    if flags["same_src"][t]:
        return "both commanded the same source cell"
    return "both moved, no shared cell"


def _divergence_detail(flags: dict, out: dict, actions: np.ndarray, t: int) -> dict:
    """Describe the first tick on which the current rule's board departs from the legacy (recorded) board.

    Everything is read off the legacy board before that tick, which is the board
    the players actually saw, plus who each rule let resolve first.
    """
    d = {"div_legacy_first_mover": -1, "div_current_first_mover": -1, "div_chaser": -1,
         "div_chaser_moves_army": -1, "div_leaver_stack": -1, "div_target_is_general": False,
         "div_detail": "none"}
    if t < 0:
        return d
    lf, cf = int(out["leg_first_mover"][t]), int(out["cur_first_mover"][t])
    d["div_legacy_first_mover"], d["div_current_first_mover"] = lf, cf
    kind = _divergence_kind(flags, t)
    src_army = out["leg_src_army"][t]
    dst_owner = out["leg_dst_owner"][t]
    dst_army = out["leg_dst_army"][t]
    dst_gen = out["leg_dst_is_general"][t]
    dst_castle = out["leg_dst_is_castle"][t]
    split = actions[t, :, 4]
    moved_army = np.where(split == 1, src_army // 2, src_army - 1)

    if kind == "one moved into the cell the other was leaving":
        chaser = 0 if flags["chase0"][t] else 1
        leaver = 1 - chaser
        d["div_chaser"] = chaser
        d["div_chaser_moves_army"] = int(moved_army[chaser])
        d["div_leaver_stack"] = int(src_army[leaver])
        d["div_target_is_general"] = bool(dst_gen[chaser])
        # legacy let the leaver go first (else the boards would agree); current lets the chaser hit the full stack
        if moved_army[chaser] > src_army[leaver]:
            what = "chaser overruns the departing stack (leaver's move then fails)"
        else:
            what = "chaser bounces off the departing stack (legacy: took the 1 left behind)"
        if dst_gen[chaser]:
            what = "chase onto a departing GENERAL: " + what
        d["div_detail"] = what
    elif kind == "both moved into the same cell":
        owner = int(dst_owner[0])          # same cell for both
        tgt = "own" if owner in (0, 1) else "neutral"
        if owner in (0, 1):
            tgt = f"P{owner}'s cell"
        if dst_gen[0]:
            tgt += " (general)"
        elif dst_castle[0]:
            tgt += " (castle)"
        d["div_detail"] = f"same destination = {tgt}, {int(dst_army[0])} armies; movers {int(moved_army[0])} vs {int(moved_army[1])}"
        # normalise to a category without the numbers for counting
        if owner in (0, 1):
            d["div_detail"] = ("same destination: one reinforces own cell" + (" (general)" if dst_gen[0] else "")
                               + " while the other attacks it")
        else:
            d["div_detail"] = "same destination: neutral cell" + (" (castle)" if dst_castle[0] else "") + \
                              (", bigger army now resolves last" if moved_army[0] != moved_army[1] else ", equal armies")
    elif kind == "head-on: each moved into the other's source":
        d["div_detail"] = "head-on swap of sources, smaller army now resolves first"
    else:
        d["div_detail"] = kind
    return d


def analyse(prep: dict) -> dict:
    meta, actions = prep["meta"], prep["actions"]
    out = simulate(prep["grid"], actions)
    flags = interaction_flags(actions)
    row = dict(meta)
    row["n_interaction_ticks"] = int(flags["shared_cell"].sum())
    row["n_both_moved_ticks"] = int(flags["both_moved"].sum())
    row.update(_rule_metrics("cur", out, actions, meta))
    row.update(_rule_metrics("leg", out, actions, meta))

    div = np.nonzero(out["diverged"])[0]
    t_div = int(div[0]) if len(div) else -1
    row["first_divergence_turn"] = t_div
    row["divergence_kind"] = _divergence_kind(flags, t_div)
    row["divergence_shared_cell"] = bool(flags["shared_cell"][t_div]) if t_div >= 0 else False
    row.update(_divergence_detail(flags, out, actions, t_div))
    # consequence of the divergence for the game as a whole
    if row["cur_agrees"]:
        row["cur_outcome"] = "reproduced exactly"
    elif row["cur_never_ended"]:
        row["cur_outcome"] = "never ended"
    elif row["cur_ended_early"] and row["cur_winner_agrees"]:
        row["cur_outcome"] = "ended early, same winner"
    elif row["cur_ended_early"]:
        row["cur_outcome"] = "ended early, different winner"
    elif row["cur_ended_late"]:
        row["cur_outcome"] = "ended late"
    elif row["cur_ended_exact"] and row["cur_winner_agrees"]:
        row["cur_outcome"] = "same end, but illegal moves along the way"
    else:
        row["cur_outcome"] = "same tick, different winner"
    fi = row["cur_first_illegal_turn"]
    row["ticks_from_divergence_to_first_illegal"] = (fi - t_div) if (fi >= 0 and t_div >= 0) else -1

    # heuristic without the legacy oracle: is there an interacting tick on or just before the first
    # illegal move? Window [fi-k, fi] inclusive — the clash usually happens on the tick of the
    # illegal move itself (the leaver's move fails in the same tick the chaser took its stack).
    for k in (0, 1, 2, 5, 10):
        if fi >= 0:
            lo = max(0, fi - k)
            row[f"cur_interaction_within_{k}_before_first_illegal"] = bool(flags["shared_cell"][lo:fi + 1].any())
        else:
            row[f"cur_interaction_within_{k}_before_first_illegal"] = False
    return row


# --------------------------------------------------------------------------- #
# Workers
# --------------------------------------------------------------------------- #
def _worker_init(cache_dir: str):
    import jax

    os.makedirs(cache_dir, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", cache_dir)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)


def _process_range(args) -> tuple[list[dict], list[dict]]:
    path, lo, hi = args
    rows_out, excluded = [], []
    for row in load_rows(path, lo, hi):
        try:
            prep = prepare(row)
        except Exception as e:  # noqa: BLE001 - keep going, report the replay
            excluded.append({"id": row.get("id"), "reason": f"unparseable: {type(e).__name__}: {e}"})
            continue
        if prep["exclude"]:
            excluded.append({"id": prep["meta"]["id"], "reason": prep["exclude"]})
            continue
        rows_out.append(analyse(prep))
    return rows_out, excluded


# --------------------------------------------------------------------------- #
# Self-test: play a few replays through plain game.step and compare
# --------------------------------------------------------------------------- #
def selftest(n: int):
    import jax.numpy as jnp

    from generals.core import game

    path = parquet_path()
    for row in load_rows(path, 0, n):
        prep = prepare(row)
        assert prep["exclude"] is None, prep
        meta, actions, grid = prep["meta"], prep["actions"], prep["grid"]
        scan = analyse(prep)
        print(f"\n== {meta['id']} v{meta['version']} {meta['map_w']}x{meta['map_h']} "
              f"moves={meta['n_moves']} last turn={meta['recorded_end_turn']} recorded winner=P{meta['recorded_winner']}")
        for legacy in (False, True):
            s = game.create_initial_state(jnp.asarray(grid))
            end_turn, end_winner = -1, -1
            first_move_note = None
            for t in range(actions.shape[0]):
                assert int(s.time) == t
                a = jnp.asarray(actions[t])
                if first_move_note is None and (actions[t, :, 0] == 0).any():
                    p = int(np.argmax(actions[t, :, 0] == 0))
                    si, sj = actions[t, p, 1], actions[t, p, 2]
                    army = int(s.armies[si, sj])
                    first_move_note = (f"first move: tick {t} P{p} from ({si},{sj}) army={army} "
                                       f"(expected 1 + t//2 = {1 + t // 2}) owned={bool(s.ownership[p, si, sj])}")
                s, info = game.step(s, a, legacy_move_priority=legacy)
                if end_turn < 0 and int(info.winner) >= 0:
                    end_turn, end_winner = t, int(info.winner)
                if t in (49, 50, 99, 100) and t < actions.shape[0]:
                    pass
            pre = "leg" if legacy else "cur"
            print(f"  [{pre}] {first_move_note}")
            print(f"  [{pre}] plain game.step: ended tick {end_turn} winner P{end_winner} | "
                  f"scan: ended tick {scan[pre + '_sim_end_turn']} winner P{scan[pre + '_sim_winner']} "
                  f"illegal={scan[pre + '_n_illegal']} first_illegal={scan[pre + '_first_illegal_turn']}")
            assert end_turn == scan[pre + "_sim_end_turn"] and end_winner == scan[pre + "_sim_winner"], \
                "scan replication of game.step disagrees with game.step"
            arm = simulate(grid, actions)["cur_army"]
            print(f"  [{pre}] P0/P1 total army at ticks 2,50,100,{actions.shape[0]-1} (current rule): "
                  + " ".join(f"t{t}={arm[t].tolist()}" for t in (2, 50, 100, actions.shape[0] - 1) if t < len(arm)))
        print(f"  first divergence (cur vs leg): tick {scan['first_divergence_turn']} [{scan['divergence_kind']}]")


# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #
def _version_block(rows: list[dict]) -> dict:
    """Agreement counts for one replay-format version (the archive holds versions 5 and 13)."""
    block = {"games": len(rows), "recorded_moves": int(sum(r["n_moves"] for r in rows))}
    for pre in ("cur", "leg"):
        for k in ("agrees", "ended_exact", "ended_early", "never_ended"):
            block[f"{pre}_{k}"] = int(sum(1 for r in rows if r[f"{pre}_{k}"]))
        block[f"{pre}_n_illegal"] = int(sum(r[f"{pre}_n_illegal"] for r in rows))
        block[f"{pre}_games_with_illegal"] = int(sum(1 for r in rows if r[f"{pre}_n_illegal"] > 0))
    return block


def summarise(rows: list[dict], excluded: list[dict], runtime_s: float, n_total: int) -> dict:
    def cnt(key):
        return int(sum(1 for r in rows if r[key]))

    def top(counter, n=12):
        return [{"kind": k, "games": v} for k, v in counter.most_common(n)]

    summary = {
        "dataset": {"replays_in_parquet": n_total, "games_used": len(rows), "games_excluded": len(excluded),
                    "excluded_by_reason": dict(Counter(e["reason"] for e in excluded)),
                    "by_format_version": dict(Counter(r["version"] for r in rows))},
        "by_format_version": {str(v): _version_block([r for r in rows if r["version"] == v])
                              for v in sorted(set(r["version"] for r in rows))},
        "moves": {"recorded_moves": int(sum(r["n_moves"] for r in rows)),
                  "dropped_duplicates": int(sum(r["n_dropped_duplicate"] for r in rows)),
                  "split_moves": int(sum(r["n_split_moves"] for r in rows)),
                  "ticks_simulated": int(sum(r["n_ticks"] for r in rows)),
                  "ticks_where_both_moved": int(sum(r["n_both_moved_ticks"] for r in rows)),
                  "ticks_where_moves_shared_a_cell": int(sum(r["n_interaction_ticks"] for r in rows)),
                  "recorded_end_is_capture": cnt("recorded_end_is_capture")},
        "runtime_seconds": round(runtime_s, 1),
    }
    for pre, name in (("cur", "current_rule"), ("leg", "legacy_rule")):
        n_ill = int(sum(r[f"{pre}_n_illegal"] for r in rows))
        summary[name] = {
            "games_fully_reproduced": cnt(f"{pre}_agrees"),
            "games_with_zero_illegal_moves": int(sum(1 for r in rows if r[f"{pre}_n_illegal"] == 0)),
            "games_with_illegal_moves": int(sum(1 for r in rows if r[f"{pre}_n_illegal"] > 0)),
            "illegal_moves": n_ill,
            "illegal_moves_by_reason": {"source not owned": int(sum(r[f"{pre}_n_illegal_not_owned"] for r in rows)),
                                        "source has <2 armies": int(sum(r[f"{pre}_n_illegal_too_few"] for r in rows)),
                                        "destination blocked": int(sum(r[f"{pre}_n_illegal_blocked"] for r in rows))},
            "illegal_moves_prestep_definition": int(sum(r[f"{pre}_n_illegal_prestep"] for r in rows)),
            "illegal_moves_from_a_source_the_mover_never_held": int(sum(r[f"{pre}_n_illegal_source_never_held"] for r in rows)),
            "games_with_such_a_move": int(sum(1 for r in rows if r[f"{pre}_n_illegal_source_never_held"] > 0)),
            "moves_after_simulated_end": int(sum(r[f"{pre}_n_moves_after_end"] for r in rows)),
            "ended_same_turn_pm1": cnt(f"{pre}_ended_same_turn"),
            "ended_exact_turn": cnt(f"{pre}_ended_exact"),
            "ended_exact_turn_same_winner": int(sum(1 for r in rows if r[f"{pre}_ended_exact"] and r[f"{pre}_winner_agrees"])),
            "ended_early": cnt(f"{pre}_ended_early"),
            "ended_late": cnt(f"{pre}_ended_late"),
            "never_ended": cnt(f"{pre}_never_ended"),
            "winner_agrees": cnt(f"{pre}_winner_agrees"),
            "winner_disagrees_among_ended": int(sum(1 for r in rows if r[f"{pre}_sim_end_turn"] >= 0 and not r[f"{pre}_winner_agrees"])),
            "first_illegal_reason": dict(Counter(r[f"{pre}_first_illegal_reason"] for r in rows if r[f"{pre}_first_illegal_turn"] >= 0)),
        }
    dis = [r for r in rows if not r["cur_agrees"]]
    div = [r for r in dis if r["first_divergence_turn"] >= 0]
    summary["current_rule_divergence"] = {
        "disagreeing_games": len(dis),
        "with_a_divergence_from_legacy_sim": len(div),
        "first_divergence_kind": dict(Counter(r["divergence_kind"] for r in div)),
        "first_divergence_on_tick_where_moves_shared_a_cell": int(sum(1 for r in div if r["divergence_shared_cell"])),
        "ticks_from_divergence_to_first_illegal_percentiles": (
            {str(q): float(np.percentile([r["ticks_from_divergence_to_first_illegal"] for r in div
                                          if r["ticks_from_divergence_to_first_illegal"] >= 0], q))
             for q in (0, 25, 50, 75, 90, 100)} if div else {}),
        "heuristic_interaction_before_first_illegal": {
            f"within_{k}_ticks": int(sum(1 for r in dis if r[f"cur_interaction_within_{k}_before_first_illegal"]))
            for k in (0, 1, 2, 5, 10)},
        "disagreeing_games_with_an_illegal_move": int(sum(1 for r in dis if r["cur_first_illegal_turn"] >= 0)),
        "outcome": dict(Counter(r["cur_outcome"] for r in dis)),
        "first_divergence_detail": dict(Counter(r["div_detail"] for r in dis).most_common()),
        "first_divergence_detail_x_outcome": [
            {"detail": k[0], "outcome": k[1], "games": v}
            for k, v in Counter((r["div_detail"], r["cur_outcome"]) for r in dis).most_common()],
        "divergence_where_legacy_and_current_first_mover_coincide": int(
            sum(1 for r in div if r["div_legacy_first_mover"] == r["div_current_first_mover"])),
        "ids_by_outcome_sample": {
            k: [r["id"] for r in dis if r["cur_outcome"] == k][:20]
            for k in sorted(set(r["cur_outcome"] for r in dis))},
        "disagreeing_game_ids_sample": [r["id"] for r in dis[:50]],
    }
    ldis = [r for r in rows if not r["leg_agrees"]]
    summary["legacy_rule_disagreements"] = {
        "disagreeing_games": len(ldis),
        "ids": [r["id"] for r in ldis[:200]],
        "first_illegal_reason": dict(Counter(r["leg_first_illegal_reason"] for r in ldis)),
    }
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
    ap.add_argument("--limit", type=int, default=None, help="only the first N replays")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--chunk", type=int, default=100, help="replays per work item")
    ap.add_argument("--selftest", type=int, default=0, help="replay N games through plain game.step and exit")
    ap.add_argument("--cache-dir", default=os.path.expanduser("~/.cache/generals-validate-jax-cache"))
    args = ap.parse_args(argv)

    _worker_init(args.cache_dir)
    if args.selftest:
        selftest(args.selftest)
        return 0

    path = parquet_path()
    import pyarrow.parquet as pq

    n_total = pq.ParquetFile(path).metadata.num_rows
    n = n_total if args.limit is None else min(args.limit, n_total)
    work = [(path, lo, min(lo + args.chunk, n)) for lo in range(0, n, args.chunk)]

    t0 = _time.time()
    rows, excluded = [], []
    if args.workers > 1:
        ctx = mp.get_context("spawn")
        with ctx.Pool(args.workers, initializer=_worker_init, initargs=(args.cache_dir,)) as pool:
            for i, (r, e) in enumerate(pool.imap_unordered(_process_range, work)):
                rows += r
                excluded += e
                done = sum(min(hi, n) - lo for _, lo, hi in work[: i + 1])
                print(f"\r{len(rows) + len(excluded)}/{n} replays  {_time.time() - t0:.0f}s", end="", file=sys.stderr)
    else:
        for w in work:
            r, e = _process_range(w)
            rows += r
            excluded += e
            print(f"\r{len(rows) + len(excluded)}/{n} replays  {_time.time() - t0:.0f}s", end="", file=sys.stderr)
    print(file=sys.stderr)
    runtime = _time.time() - t0
    rows.sort(key=lambda r: r["id"])

    os.makedirs(args.out, exist_ok=True)
    if rows:
        with open(os.path.join(args.out, "per_game.csv"), "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            wr.writeheader()
            wr.writerows(rows)
    with open(os.path.join(args.out, "excluded.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["id", "reason"])
        wr.writeheader()
        wr.writerows(excluded)
    summary = summarise(rows, excluded, runtime, n)
    summary["engine_commit"] = os.popen(f"git -C {os.path.dirname(os.path.abspath(__file__))} rev-parse HEAD").read().strip()
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("legacy_rule_disagreements",)}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
