#!/usr/bin/env python
"""Free-for-all and 2v2: compare our engine with generals.io's own replay engine tile by tile, tick by tick.

Same method as ../board_diff.py, for N players and teams. The site's boards come from site_engine/run_many.js
(owners[t], armies[t] for t = 0..turns; entry t is the board after the moves recorded with turn t-1,
i.e. our state at the start of tick t).

Server-side events (row["afks"], one {index, turn} per event; bundle module 3067, processReplayAfks, run at the
start of nextTurn before that turn's moves):
  first event for a player  -> killPlayer: the player is dead (moves cleared, later moves rejected), alivePlayers--,
                               but the land, general and cities stay theirs and keep growing; the general can still be
                               captured (executePlayerCapture hands the land to the capturer at Math.round(0.5 * army));
  second event              -> tryNeutralizePlayer: if the general tile is still theirs, every tile they own goes to the
                               first living teammate (team games) or becomes neutral (no teammate / FFA), armies unchanged
                               (replaceAll(p, q, 1)), and the general becomes a city; if the general has fallen: no-op.
  The game ends when one player (one team) is alive (isOver); that turn's moves and income still run.
Our engine has no AFK concept: it plays every recorded move and ends only by capture. Runs per game:
  engine        the record alone                                          -> agreement up to the first event
  engine_afk    the same scan with the site events applied to the state at the start of the tick, outside game.step
                (kill = mark eliminated; neutralize = cells -> teammate or neutral, general -> castle) -> full-game agreement
  engine_afk_team (team games) events plus two team rules of the site's Map.attack / checkAttackValid that the engine
                lacks, applied outside game.step: (1) a move onto a teammate's GENERAL pools the armies but the tile stays
                the teammate's (restored after the step); (2) a move from a tile with 1 army onto a teammate's ordinary
                tile is legal and hands the tile to the mover with no army moved (applied before the step).
  engine_patched  (--patched, separate process) events plus the three site rules monkeypatched into the engine's functions
                for this process (apply_site_team_patches: ally general keeps its owner; 1-army ally transfer; general
                flag in the move order) -- the engine change that would make the record + events replay exactly.
Usage: python board_diff_ffa.py <siteboards dir> 'gior/*.gior' [--out f] [--patched]
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
    actions (T, n_pad, 5); afk_events (T, n_pad) = number of site AFK events for that player at that tick;
    teams (n_pad,) team ids (the replay's `teams` when present, else arange; padded players get their own teams)."""
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
    special = set(cities) | set(mountains) | set(generals)
    if len(special) != len(cities) + len(mountains) + len(generals): return {"exclude": "overlapping special tiles", "meta": meta}
    teams_raw = row.get("teams")
    if teams_raw:
        if len(teams_raw) != N: return {"exclude": "teams length", "meta": meta}
        ids = {t: i for i, t in enumerate(dict.fromkeys(teams_raw))}
        teams = np.array([ids[t] for t in teams_raw] + list(range(len(ids), len(ids) + n_pad - N)), dtype=np.int32)
    else:
        teams = np.arange(n_pad, dtype=np.int32)
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
                first_afk_turn=min([a["turn"] for a in row["afks"]], default=-1), teams=teams[:N].tolist(), team_game=bool(teams_raw))
    return {"exclude": None, "grid": grid, "actions": actions, "afk": afk, "teams": teams, "meta": meta}


def _afk_event(s, p):
    """One site AFK event for player p on state s: kill if alive, neutralize (or hand to the first living teammate) if dead."""
    N = s.ownership.shape[0]

    def kill(s):
        return s._replace(eliminated=s.eliminated.at[p].set(True))

    def neutralize(s):
        gp = s.general_positions[p]
        still_mine = s.ownership[p, gp[0], gp[1]] & s.generals[gp[0], gp[1]]      # tryNeutralizePlayer: only while the general tile is theirs
        cells = s.ownership[p] & still_mine
        mates = (s.teams == s.teams[p]) & ~s.eliminated & (jnp.arange(N) != p)
        has_mate = jnp.any(mates); q = jnp.argmax(mates)
        onehot = (jnp.arange(N) == q) & has_mate
        ownership = jnp.where(cells[None], onehot[:, None, None], s.ownership)
        return s._replace(ownership=ownership, ownership_neutral=s.ownership_neutral | (cells & ~has_mate),
                          castles=s.castles | (s.generals & cells), generals=s.generals & ~cells)
    return jax.lax.cond(s.eliminated[p], neutralize, kill, s)


def _team_rule_1army(s, a):
    """Site checkAttackValid + Map.attack: a move from a 1-army tile onto a teammate's ordinary tile hands the tile
    to the mover with no army. Returns the state with those transfers applied and the actions with them passed."""
    N = a.shape[0]; H, W = s.armies.shape
    si, sj = jnp.clip(a[:, 1], 0, H - 1), jnp.clip(a[:, 2], 0, W - 1)
    di, dj = jnp.clip(si + game.DIRECTIONS[a[:, 3], 0], 0, H - 1), jnp.clip(sj + game.DIRECTIONS[a[:, 3], 1], 0, W - 1)
    idx = jnp.arange(N)
    dest_owner = jnp.where(s.ownership[:, di, dj].any(0), s.ownership[:, di, dj].argmax(0), -1)      # (N,)
    same_team = (dest_owner >= 0) & (s.teams[jnp.clip(dest_owner, 0)] == s.teams) & (dest_owner != idx)
    hit = (a[:, 0] == 0) & s.ownership[idx, si, sj] & (s.armies[si, sj] == 1) & same_team & ~s.generals[di, dj] & ~s.eliminated
    ownership = s.ownership
    for p in range(N):
        ownership = jnp.where(hit[p], ownership.at[:, di[p], dj[p]].set(idx == p), ownership)
    a = jnp.where(hit[:, None], jnp.asarray(PASS)[None, :], a)
    return s._replace(ownership=ownership), a, hit.sum()


def _team_rule_general(s_before, s_after):
    """Site Map.attack: a merge onto a teammate's general leaves the tile with the teammate; restore it after the step."""
    N = s_after.ownership.shape[0]
    own_b = jnp.where(s_before.ownership.any(0), s_before.ownership.argmax(0), -1)
    own_a = jnp.where(s_after.ownership.any(0), s_after.ownership.argmax(0), -1)
    fix = s_after.generals & s_before.generals & (own_b >= 0) & (own_a >= 0) & (own_a != own_b) \
        & (s_after.teams[jnp.clip(own_a, 0)] == s_after.teams[jnp.clip(own_b, 0)])
    onehot = jnp.arange(N)[:, None, None] == own_b[None]
    return s_after._replace(ownership=jnp.where(fix[None], onehot, s_after.ownership)), fix.sum()


# ----------------------------------------------------------------------------------------------------------------------
# --patched: the three team-play rules of the site that the engine lacks, applied as monkeypatches of generals.core.game
# in this process only (the engine's files are untouched). Site code: bundle module 5124 Map.attack, 3067 checkAttackValid,
# 7155 MoveResolver.determineMoveOrder.
# ----------------------------------------------------------------------------------------------------------------------
def apply_site_team_patches():
    import inspect
    orig_apply_move, orig_execute_move = game._apply_move, game._execute_move

    def _apply_move_site(state, player_idx, si, sj, di, dj, army_to_move, spoils=True):
        # Map.attack: a friendly move onto a teammate's GENERAL pools the armies but the tile stays the teammate's
        # ("a !== s && i[a] !== t && this.setTile(t, s)")
        target_owners = state.ownership[:, di, dj]
        same_team = state.teams == state.teams[player_idx]
        keep = jnp.any(target_owners & same_team) & ~target_owners[player_idx] & state.generals[di, dj]
        s = orig_apply_move(state, player_idx, si, sj, di, dj, army_to_move, spoils)
        return s._replace(ownership=jnp.where(keep, s.ownership.at[:, di, dj].set(target_owners), s.ownership))

    def _execute_move_site(state, player_idx, si, sj, direction, split_army, spoils=True):
        # checkAttackValid: a move from a 1-army tile is valid when the destination is a teammate's tile
        # ("1 !== armyAt(t) || teams[e] === teams[tileAt(n)] && tileAt(n) !== e"); Map.attack then moves 0 armies and
        # hands the tile to the mover unless it is the teammate's general.
        H, W = state.armies.shape
        di = si + game.DIRECTIONS[direction, 0]; dj = sj + game.DIRECTIONS[direction, 1]
        in_b = (si >= 0) & (si < H) & (sj >= 0) & (sj < W) & (di >= 0) & (di < H) & (dj >= 0) & (dj < W)
        cdi, cdj = jnp.clip(di, 0, H - 1), jnp.clip(dj, 0, W - 1); csi, csj = jnp.clip(si, 0, H - 1), jnp.clip(sj, 0, W - 1)
        target_owners = state.ownership[:, cdi, cdj]
        same_team = state.teams == state.teams[player_idx]
        ally_tile = jnp.any(target_owners & same_team) & ~target_owners[player_idx]
        transfer = (in_b & state.ownership[player_idx, csi, csj] & (state.armies[csi, csj] == 1) & ally_tile
                    & ~state.eliminated[player_idx] & state.passable[cdi, cdj] & ~state.generals[cdi, cdj])
        s = orig_execute_move(state, player_idx, si, sj, direction, split_army, spoils)
        mover = jnp.arange(state.ownership.shape[0]) == player_idx
        return s._replace(ownership=jnp.where(transfer, s.ownership.at[:, cdi, cdj].set(mover), s.ownership))

    # determineMoveOrder: isGeneralAttack = "the destination is a general tile", friendly merges included
    src = inspect.getsource(game._determine_move_order)
    old = "general_attack = state.generals[cdi, cdj] & ~defensive & ~passes"
    assert src.count(old) == 1
    ns = dict(vars(game)); exec(src.replace(old, "general_attack = state.generals[cdi, cdj] & ~passes"), ns)
    game._apply_move, game._execute_move, game._determine_move_order = _apply_move_site, _execute_move_site, ns["_determine_move_order"]


PATCHED = "--patched" in sys.argv
if PATCHED:
    sys.argv.remove("--patched"); apply_site_team_patches()


@partial(jax.jit, static_argnames=("n_players", "afk_model", "team_rules"))
def boards(grid, actions, afk, teams, n_players, afk_model, team_rules):
    s0 = game.create_initial_state(grid, teams=teams)

    def tick(s, inp):
        a, ev = inp
        own0, arm0 = s.ownership, s.armies                    # the board the site dumps before this turn (before processReplayAfks)
        if afk_model:
            for p in range(n_players):
                for rep in range(2):
                    s = jax.lax.cond(ev[p] > rep, lambda s: _afk_event(s, p), lambda s: s, s)
        n1 = n2 = jnp.int32(0)
        if team_rules:
            s, a, n1 = _team_rule_1army(s, a)
        si, sj = a[:, 1], a[:, 2]
        H, W = s.armies.shape; idx = jnp.arange(n_players)
        di, dj = jnp.clip(si + game.DIRECTIONS[a[:, 3], 0], 0, H - 1), jnp.clip(sj + game.DIRECTIONS[a[:, 3], 1], 0, W - 1)
        dest = s.ownership[:, di, dj]; ally_dest = jnp.any(dest & (s.teams[:, None] == s.teams[None, :]), axis=0) & ~dest[idx, idx]
        # the site's checkAttackValid: mover owns the source with >= 2 armies, or with 1 army when the destination is a teammate's tile
        legal = (a[:, 0] != 0) | (s.ownership[idx, si, sj] & ((s.armies[si, sj] >= 2) | ((s.armies[si, sj] == 1) & ally_dest)) & ~s.eliminated)
        s2, info = game.step(s, a, general_trade=True)
        if team_rules:
            s2, n2 = _team_rule_general(s, s2)
        return s2, (own0, arm0, legal, info.winner, s2.eliminated, n1, n2)

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
    teams = prep["teams"]
    res = dict(id=row["id"], n_players=N, teams=meta["teams"], team_game=meta["team_game"], turns=int(T - 1), site_turns=int(site["turns"]),
               site_winner=site["winner"], consumed=site["consumed"], n_moves=site["n_moves"], n_afk_events=meta["n_afk_events"],
               first_afk_turn=meta["first_afk_turn"], map=[W, H])
    runs = [("engine_patched", True, False)] if PATCHED else [("engine", False, False), ("engine_afk", True, False)] + ([("engine_afk_team", True, True)] if meta["team_game"] else [])
    for name, model, trules in runs:
        own, arm, legal, winner, elim, n1, n2 = boards(jnp.asarray(prep["grid"]), jnp.asarray(acts), jnp.asarray(afk), jnp.asarray(teams), NP, model, trules)
        own = np.asarray(own)[:T, :, :H, :W].reshape(T, NP, -1); arm = np.asarray(arm)[:T, :H, :W].reshape(T, -1)
        oo = np.where(own.any(1), own.argmax(1), -1)
        bad = (oo != so_) | (arm != sa[:T])
        in_game = bad[:-1]                                                            # the site still pays that turn's income after the final capture
        first = int(np.argmax(in_game.any(1))) if in_game.any() else -1
        legal, winner, elim = np.asarray(legal)[:T - 1, :N], np.asarray(winner)[:T - 1], np.asarray(elim)[:T - 1, :N]
        tm = np.array(meta["teams"]); alive_teams = np.array([len(set(tm[~e])) for e in elim])
        ends = np.nonzero((alive_teams <= 1) | (winner >= 0))[0]
        end_tick = int(ends[0]) if len(ends) else -1                                  # the tick whose step leaves one team alive
        our_winner = sorted(int(p) for p in np.nonzero(tm == tm[np.argmax(~elim[end_tick])])[0]) if end_tick >= 0 and alive_teams[end_tick] == 1 else []
        first_afk = meta["first_afk_turn"]
        pre_afk_ok = bool(not in_game[:first_afk + 1].any()) if first_afk >= 0 else bool(not in_game.any())   # boards 0..first_afk: the event lands after board first_afk was dumped
        res[name] = dict(mismatch_ticks=int(in_game.any(1).sum()), first_mismatch=first,
                         first_tiles=[[int(i), int(so_[first, i]), int(sa[first, i]), int(oo[first, i]), int(arm[first, i])] for i in np.nonzero(bad[first])[0][:6]] if first >= 0 else [],
                         terminal_mismatch=bool(bad[-1].any()), identical_before_first_afk=pre_afk_ok,
                         rejected_moves=int((~legal).sum()), end_tick=end_tick, our_winner=our_winner,
                         rule_1army_transfers=int(np.asarray(n1)[:T - 1].sum()), rule_ally_general_merges=int(np.asarray(n2)[:T - 1].sum()),
                         same_end=bool(end_tick + 1 == site["turns"] and sorted(site["winner"]) == our_winner))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("site_dir"); ap.add_argument("pattern"); ap.add_argument("--out")
    a = ap.parse_args(); res = [run(f, a.site_dir) for f in sorted(glob.glob(a.pattern))]
    ok = [r for r in res if "excluded" not in r]
    print(f"games {len(ok)} (excluded {len(res)-len(ok)}: {[r['excluded'] for r in res if 'excluded' in r]}); ticks {sum(r['turns'] for r in ok)}; moves {sum(r['n_moves'] for r in ok)}; games with AFK events {sum(r['n_afk_events']>0 for r in ok)}")
    for name in ("engine", "engine_afk", "engine_afk_team", "engine_patched"):
        rr = [r[name] for r in ok if name in r]
        if not rr: continue
        print(f"[{name}] identical at every in-game tick {sum(x['mismatch_ticks']==0 for x in rr)}; identical up to the first AFK event {sum(x['identical_before_first_afk'] for x in rr)}; "
              f"same end (turn and winner) {sum(x['same_end'] for x in rr)}; terminal board differs {sum(x['terminal_mismatch'] for x in rr)}; recorded moves rejected {sum(x['rejected_moves'] for x in rr)}"
              + (f"; 1-army transfers to a teammate {sum(x['rule_1army_transfers'] for x in rr)}; merges onto a teammate's general {sum(x['rule_ally_general_merges'] for x in rr)}" if name == 'engine_afk_team' else ''))
    final = "engine_patched" if PATCHED else ("engine_afk_team" if any("engine_afk_team" in r for r in ok) else "engine_afk")
    for r in ok:
        x = r[final]
        if x["mismatch_ticks"] or not x["same_end"]:
            print(f"  DIFF {r['id']} N={r['n_players']} afks={r['n_afk_events']} first_afk={r['first_afk_turn']} first mismatch tick {x['first_mismatch']} tiles[idx,site_o,site_a,our_o,our_a] {x['first_tiles']} "
                  f"mismatching {x['mismatch_ticks']}/{r['turns']} end ours {x['end_tick']}+1 w{x['our_winner']} site {r['site_turns']} w{r['site_winner']}")
    if a.out: json.dump(res, open(a.out, "w"))
