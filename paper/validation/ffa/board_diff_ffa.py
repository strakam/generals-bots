#!/usr/bin/env python
"""Free-for-all: compare our engine with generals.io's own replay engine tile by tile, tick by tick.

Same method as ../board_diff.py, for N players. The site's boards come from site_engine/run_many.js
(owners[t], armies[t] for t = 0..turns; entry t is the board after the moves recorded with turn t-1,
i.e. our state at the start of tick t).

AFK / surrender events (row["afks"], one {index, turn} per event). What the site's Game does
(bundle module 3067, processReplayAfks, run at the start of nextTurn before that turn's moves):
  first event for a player  -> killPlayer: the player is dead (moves cleared, later moves rejected),
                               alivePlayers--, but the land, general and cities stay theirs and keep
                               growing; the general can still be captured (executePlayerCapture hands
                               the land to the capturer at Math.round(0.5 * army));
  second event (50 turns on) -> tryNeutralizePlayer: if the general tile is still theirs, every tile
                               they own becomes neutral with its army unchanged (replaceAll(p, -1)) and
                               the general becomes a city; if the general has fallen meanwhile: no-op.
  The game ends when one player is alive (killPlayer -> isOver); that turn's moves and income still run.
Our engine has no AFK concept: it plays every recorded move and ends only by capture. Two runs:
  engine     the record alone (AFK events ignored)                    -> agreement up to the first AFK event
  engine+afk the same scan with the two site events applied to the state at the start of the tick,
             outside game.step (kill = mark eliminated; neutralize = cells -> neutral, general -> castle)
             -> full-game agreement.
Usage: python board_diff_ffa.py <siteboards dir> 'gior/*.gior' [--out f]
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import argparse, glob, json, sys  # noqa: E402
import numpy as np  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gior  # noqa: E402
import replay_prep as rp  # noqa: E402
import jax, jax.numpy as jnp  # noqa: E402
from functools import partial  # noqa: E402
from generals.core import game  # noqa: E402

PAD = 36          # FFA maps in the sample go up to 33 x 33
NP = 8            # every game is padded to this many players (absent players start eliminated: no general on the grid)
BUCKET = 512
PASS = np.array(rp.PASS, dtype=np.int32)


def prepare(row: dict, n_pad: int = NP, pad: int = PAD) -> dict:
    """N-player variant of replay_prep.prepare: grid values 1..N are the generals, cities > N;
    actions (T, n_pad, 5); afk_events (T, n_pad) = number of site AFK events for that player at that tick."""
    w, h = int(row["mapWidth"]), int(row["mapHeight"])
    generals = list(row["generals"] or []); N = len(row["usernames"] or [])
    meta = {"id": row["id"], "version": int(row["version"]), "map_w": w, "map_h": h, "n_players": N}
    if N < 2 or N > n_pad: return {"exclude": f"{N} players", "meta": meta}
    if len(generals) != N or any(g is None or g < 0 or g >= w * h for g in generals): return {"exclude": "generals", "meta": meta}
    if w > pad or h > pad: return {"exclude": f"map larger than {pad}", "meta": meta}
    cities, city_armies, mountains = list(row["cities"] or []), list(row["cityArmies"] or []), list(row["mountains"] or [])
    if len(cities) != len(city_armies) or any(a <= n_pad for a in city_armies): return {"exclude": "city armies", "meta": meta}
    for k in ("modifiers", "lookouts", "observatories", "tunnels", "swamps", "deserts", "strongholds", "neutrals"):
        if row.get(k) or (row.get("extras") or {}).get(k): return {"exclude": f"not the base game: {k}", "meta": meta}
    if row.get("teams"): return {"exclude": "teams", "meta": meta}
    special = set(cities) | set(mountains) | set(generals)
    if len(special) != len(cities) + len(mountains) + len(generals): return {"exclude": "overlapping special tiles", "meta": meta}
    grid = np.full((pad, pad), rp.MOUNTAIN, dtype=np.int32); board = np.zeros(h * w, dtype=np.int32)
    board[mountains] = rp.MOUNTAIN
    for t, a in zip(cities, city_armies): board[t] = int(a)
    for p, g in enumerate(generals): board[g] = p + 1
    grid[:h, :w] = board.reshape(h, w)
    moves = row["moves"] or []
    if not moves: return {"exclude": "no moves", "meta": meta}
    turns = [m[4] for m in moves]
    if turns != sorted(turns): return {"exclude": "moves not sorted", "meta": meta}
    T = max(int(turns[-1]), max([a["turn"] for a in row["afks"]] + [0])) + 1
    actions = np.tile(PASS, (T, n_pad, 1)); seen = set(); n_dup = 0
    for p, s, e, is50, t in moves:
        p, s, e, t = int(p), int(s), int(e), int(t)
        if not (0 <= p < N) or t < 0 or not (0 <= s < w * h and 0 <= e < w * h): return {"exclude": "bad move", "meta": meta}
        r0, c0 = divmod(s, w); r1, c1 = divmod(e, w); d = rp.DELTA_TO_DIR.get((r1 - r0, c1 - c0))
        if d is None: return {"exclude": "non-adjacent move", "meta": meta}
        if (p, t) in seen: n_dup += 1; continue
        seen.add((p, t)); actions[t, p] = (0, r0, c0, d, 1 if is50 else 0)
    afk = np.zeros((T, n_pad), dtype=np.int32)
    for a in row["afks"]: afk[int(a["turn"]), int(a["index"])] += 1
    meta.update(n_moves=len(moves) - n_dup, n_dropped_duplicate=n_dup, n_ticks=T, n_afk_events=len(row["afks"]),
                first_afk_turn=min([a["turn"] for a in row["afks"]], default=-1))
    return {"exclude": None, "grid": grid, "actions": actions, "afk": afk, "meta": meta}


def _afk_event(s, p):
    """One site AFK event for player p on state s: kill if alive, neutralize if already dead."""
    def kill(s):
        return s._replace(eliminated=s.eliminated.at[p].set(True))

    def neutralize(s):
        cells = s.ownership[p]
        return s._replace(ownership=s.ownership.at[p].set(False), ownership_neutral=s.ownership_neutral | cells,
                          castles=s.castles | (s.generals & cells), generals=s.generals & ~cells)
    return jax.lax.cond(s.eliminated[p], neutralize, kill, s)


@partial(jax.jit, static_argnames=("n_players", "afk_model"))
def boards(grid, actions, afk, n_players, afk_model):
    s0 = game.create_initial_state(grid, num_players=n_players)

    def tick(s, inp):
        a, ev = inp
        own0, arm0 = s.ownership, s.armies                    # the board the site dumps before this turn (before processReplayAfks)
        if afk_model:
            for p in range(n_players):
                for rep in range(2):
                    s = jax.lax.cond(ev[p] > rep, lambda s: _afk_event(s, p), lambda s: s, s)
        si, sj = a[:, 1], a[:, 2]
        legal = (a[:, 0] != 0) | (s.ownership[jnp.arange(n_players), si, sj] & (s.armies[si, sj] >= 2) & ~s.eliminated)
        s2, info = game.step(s, a, general_trade=True)
        return s2, (own0, arm0, legal, info.winner, s2.eliminated)

    _, out = jax.lax.scan(tick, s0, (actions, afk))
    return out


def run(path, site_dir):
    row = gior.row(path); prep = prepare(row)
    if prep["exclude"]: return dict(id=row["id"], excluded=prep["exclude"])
    sp = os.path.join(site_dir, row["id"] + ".json")
    if not os.path.exists(sp): return dict(id=row["id"], excluded="no site board")
    site = json.load(open(sp)); W, H = site["W"], site["H"]; meta = prep["meta"]; N = meta["n_players"]
    so, sa = np.array(site["owners"]), np.array(site["armies"])                     # (turns+1, H*W)
    T = min(so.shape[0], prep["actions"].shape[0] + 1)
    Tp = -(-T // BUCKET) * BUCKET
    acts = np.tile(PASS, (Tp, NP, 1)); acts[:prep["actions"].shape[0]] = prep["actions"]
    afk = np.zeros((Tp, NP), dtype=np.int32); afk[:prep["afk"].shape[0]] = prep["afk"]
    so_ = np.where(so[:T] < 0, -1, so[:T])                                            # site: -1 empty, -2 mountain
    res = dict(id=row["id"], n_players=N, turns=int(T - 1), site_turns=int(site["turns"]), site_winner=site["winner"],
               consumed=site["consumed"], n_moves=site["n_moves"], n_afk_events=meta["n_afk_events"],
               first_afk_turn=meta["first_afk_turn"], map=[W, H])
    for name, model in (("engine", False), ("engine_afk", True)):
        own, arm, legal, winner, elim = boards(jnp.asarray(prep["grid"]), jnp.asarray(acts), jnp.asarray(afk), NP, model)
        own = np.asarray(own)[:T, :, :H, :W].reshape(T, NP, -1); arm = np.asarray(arm)[:T, :H, :W].reshape(T, -1)
        oo = np.where(own.any(1), own.argmax(1), -1)
        bad = (oo != so_) | (arm != sa[:T])
        in_game = bad[:-1]                                                            # the site still pays that turn's income after the final capture
        first = int(np.argmax(in_game.any(1))) if in_game.any() else -1
        legal, winner, elim = np.asarray(legal)[:T - 1, :N], np.asarray(winner)[:T - 1], np.asarray(elim)[:T - 1, :N]
        alive = (~elim).sum(1); ends = np.nonzero((alive <= 1) | (winner >= 0))[0]
        end_tick = int(ends[0]) if len(ends) else -1                                  # the tick whose step leaves one player alive
        our_winner = int(np.argmax(~elim[end_tick])) if end_tick >= 0 and alive[end_tick] == 1 else (int(winner[end_tick]) if end_tick >= 0 else -1)
        first_afk = meta["first_afk_turn"]
        pre_afk_ok = bool(not in_game[:first_afk + 1].any()) if first_afk >= 0 else bool(not in_game.any())   # boards 0..first_afk: the event lands after board first_afk was dumped
        res[name] = dict(mismatch_ticks=int(in_game.any(1).sum()), first_mismatch=first,
                         first_tiles=[[int(i), int(so_[first, i]), int(sa[first, i]), int(oo[first, i]), int(arm[first, i])] for i in np.nonzero(bad[first])[0][:6]] if first >= 0 else [],
                         terminal_mismatch=bool(bad[-1].any()), identical_before_first_afk=pre_afk_ok,
                         rejected_moves=int((~legal).sum()), end_tick=end_tick, our_winner=our_winner,
                         same_end=bool(end_tick + 1 == site["turns"] and site["winner"] == [our_winner]))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("site_dir"); ap.add_argument("pattern"); ap.add_argument("--out")
    a = ap.parse_args(); res = [run(f, a.site_dir) for f in sorted(glob.glob(a.pattern))]
    ok = [r for r in res if "excluded" not in r]
    print(f"games {len(ok)} (excluded {len(res)-len(ok)}: {[r['excluded'] for r in res if 'excluded' in r]}); ticks {sum(r['turns'] for r in ok)}; moves {sum(r['n_moves'] for r in ok)}; games with AFK events {sum(r['n_afk_events']>0 for r in ok)}")
    for name in ("engine", "engine_afk"):
        rr = [r[name] for r in ok]
        print(f"[{name}] identical at every in-game tick {sum(x['mismatch_ticks']==0 for x in rr)}; identical up to the first AFK event {sum(x['identical_before_first_afk'] for x in rr)}; "
              f"same end (turn and winner) {sum(x['same_end'] for x in rr)}; terminal board differs {sum(x['terminal_mismatch'] for x in rr)}; recorded moves rejected {sum(x['rejected_moves'] for x in rr)}")
    for r in ok:
        x = r["engine_afk"]
        if x["mismatch_ticks"] or not x["same_end"]:
            print(f"  DIFF {r['id']} N={r['n_players']} afks={r['n_afk_events']} first_afk={r['first_afk_turn']} first mismatch tick {x['first_mismatch']} tiles[idx,site_o,site_a,our_o,our_a] {x['first_tiles']} "
                  f"mismatching {x['mismatch_ticks']}/{r['turns']} end ours {x['end_tick']}+1 w{x['our_winner']} site {r['site_turns']} w{r['site_winner']}")
    if a.out: json.dump(res, open(a.out, "w"))
