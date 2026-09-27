"""Turn one decoded generals.io replay (see gior.py) into engine inputs: the padded grid and the per-tick
action array. Base game only: replays with modifiers or optional map features are excluded.

Grid encoding (generals.core.game.create_initial_state): 0 empty, -2 mountain, 1/2 the generals of P0/P1,
> 2 a city with that garrison. tile = row * mapWidth + col. A move recorded with turn t is played in the
engine step taken from state.time == t. Ticks without a move for a player get a pass action.
"""
import numpy as np

PAD = 32          # every grid is padded to PAD x PAD with mountains so the JIT compiles once per tick bucket
MOUNTAIN, EMPTY = -2, 0
DELTA_TO_DIR = {(-1, 0): 0, (1, 0): 1, (0, -1): 2, (0, 1): 3}   # UP DOWN LEFT RIGHT
PASS = (1, 0, 0, 0, 0)


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
    # base game only: ladder event days with modifiers or optional map features are excluded
    for k in ("modifiers", "lookouts", "observatories", "tunnels", "swamps", "deserts", "strongholds", "neutrals"):
        if row.get(k) or (row.get("extras") or {}).get(k):
            return {"exclude": f"not the base game: {k}", "meta": meta}
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
