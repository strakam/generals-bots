"""
Edit this file to implement your agent.

`Agent.act(obs)` is called once per turn. The `obs` argument is built by
`main.py` from the wire-protocol frame and has these fields:

    obs.H, obs.W            board dimensions (constant for the whole game)
    obs.turn                current turn number, increments each step
    obs.my_land             total cells you own
    obs.my_army             total armies summed over your cells
    obs.opp_land            opponent's land count (visible at all times)
    obs.opp_army            opponent's army total (visible at all times)
    obs.type_grid[r][c]     0=fog, 1=plain, 2=mountain, 3=castle, 4=general, 5=structure-in-fog
    obs.owner_grid[r][c]    0=neutral/unknown, 1=me, 2=opp  (perspective-relative)
    obs.army_grid[r][c]     army count, 0 in fog or empty

`act` must return a 5-tuple `(pass, row, col, direction, split)`:

    pass:       0 to move, 1 to skip the turn, 2 to build a castle
    row, col:   source cell of a move (must be owned by you and have
                army > 1), or the cell to build a castle on
    direction:  0=up, 1=down, 2=left, 3=right (ignored for pass/build)
    split:      0=move all-but-one armies, 1=move half (floor division;
                ignored for pass/build)

To build a castle on one of your own plain cells, return (2, r, c, 0, 0) —
the price is paid from the army standing on (r, c) (see the rules page).

Invalid moves and builds are silently treated as a pass by the engine.
"""

# A no-op action — used when no valid move exists or as a safe default.
PASS = (1, 0, 0, 0, 0)

# (dr, dc) offsets for direction codes 0..3
DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# Castle pricing, mirrored from the engine (generals/modifiers/build_castles.py).
# A cell costs BASE_COST, plus max(0, PENALTY - DECAY * d) for every structure
# you already own at manhattan distance d — so crowding your own general or
# castles gets expensive fast, and anything 5+ cells away is free.
BASE_COST = 35
PROXIMITY_PENALTY = 10
PROXIMITY_DECAY = 2

# Keep this much army on the new castle after paying for it. The price comes
# out of the army standing on the cell, so building at exactly cost leaves it
# at 0 — a single enemy unit takes it.
BUILD_RESERVE = 8

# Don't build within this manhattan distance of a visible enemy cell: near the
# frontier that army is worth more as a fighting stack than as economy.
BUILD_SAFE_DISTANCE = 3

# A castle earns +1 army every even tick (0.5/turn); land earns +1 per cell only
# every 50 ticks (0.02/turn). One castle is therefore worth ~25 land, and repays
# its ~40 army in ~80 turns. In a 1200-turn match that stays profitable almost to
# the end, so build until there is no time left to recoup the cost.
LAST_BUILD_TURN = 1050

# No meaningful cap: with a 25:1 income edge over land, every castle we can
# safely afford is worth building. The real limiters are the proximity
# surcharge on the price and how much army we can spare.
MAX_CASTLES = 99

# Ceiling on how much we let the general hoard while saving. Above the dearest
# a castle can cost (base + the full adjacency surcharge, plus the reserve and
# the one army that stays behind), a general with no buildable neighbour is
# never going to spend it — so hand the stack back to the expander.
HOLD_CEILING = BASE_COST + PROXIMITY_PENALTY + BUILD_RESERVE + 1


def _is_passable(t):
    # Mountains (2) and fogged-structures (5) are impassable; everything else
    # can be entered (including fog — you just don't know what's there).
    return t != 2 and t != 5


class Agent:
    """Expander strategy, plus castle building.

    Each turn, first consider building a castle: on an owned plain cell that
    can pay the price and still keep BUILD_RESERVE army, well away from the
    enemy, cheapest cell first. Castles are the only economy in the
    build-castles ruleset — maps spawn with no neutral ones — so an idle stack
    sitting on 40+ army is better spent on permanent income.

    Otherwise fall back to the expander move: maximize
        score = src_army * (10 if expansion else 1) * (2 if opponent else 1)
    among captures (src_army > dest_army + 1). If no capture is possible
    but some legal move exists, take the first one. Otherwise pass.
    """

    def __init__(self, player_id, H, W):
        # We get the static game info once at startup. You don't have to
        # store any of it on the agent if you don't want to.
        self.player_id = player_id
        self.H = H
        self.W = W

    def _build_cost(self, r, c, structures):
        """Price of a castle at (r, c), matching the engine's cost grid."""
        cost = BASE_COST
        for sr, sc in structures:
            surcharge = PROXIMITY_PENALTY - PROXIMITY_DECAY * (abs(sr - r) + abs(sc - c))
            if surcharge > 0:
                cost += surcharge
        return cost

    def _survey(self, obs):
        """One pass over the board: our structures, visible enemies, general."""
        structures = []
        enemies = []
        general = None
        for r in range(obs.H):
            for c in range(obs.W):
                owner = obs.owner_grid[r][c]
                if owner == 1:
                    t = obs.type_grid[r][c]
                    if t == 4:
                        general = (r, c)
                        structures.append((r, c))
                    elif t == 3:
                        structures.append((r, c))
                elif owner == 2:
                    enemies.append((r, c))
        return structures, enemies, general

    def _is_safe(self, r, c, enemies):
        return not any(abs(er - r) + abs(ec - c) <= BUILD_SAFE_DISTANCE
                       for er, ec in enemies)

    def _find_build(self, obs, structures, enemies):
        """Cheapest affordable, safe castle site, or None.

        Only cells we own and that are plain (type 1) are buildable — the
        engine rejects builds on our general or an existing castle.
        """
        best = None
        best_cost = None
        for r in range(obs.H):
            for c in range(obs.W):
                if obs.owner_grid[r][c] != 1 or obs.type_grid[r][c] != 1:
                    continue
                army = obs.army_grid[r][c]
                # Cheap reject before the per-structure cost loop.
                if army < BASE_COST + BUILD_RESERVE:
                    continue
                if not self._is_safe(r, c, enemies):
                    continue
                cost = self._build_cost(r, c, structures)
                if army < cost + BUILD_RESERVE:
                    continue
                if best_cost is None or cost < best_cost:
                    best_cost = cost
                    best = (2, r, c, 0, 0)
        return best

    def _stage_general(self, obs, structures, enemies, general):
        """Step the general's savings onto an adjacent cell we can build on.

        The general is the only tile that accumulates on its own, but it can't
        be built on — so once it holds enough to fund a neighbouring castle
        (plus the one army that always stays behind), walk the stack one cell
        out. `_find_build` picks it up on the following turn.
        """
        gr, gc = general
        g_army = obs.army_grid[gr][gc]

        best = None
        best_cost = None
        for d, (dr, dc) in enumerate(DIRECTIONS):
            nr, nc = gr + dr, gc + dc
            if not (0 <= nr < obs.H and 0 <= nc < obs.W):
                continue
            # Must already be ours and plain, so the whole stack arrives intact.
            if obs.owner_grid[nr][nc] != 1 or obs.type_grid[nr][nc] != 1:
                continue
            if not self._is_safe(nr, nc, enemies):
                continue
            cost = self._build_cost(nr, nc, structures)
            # -1 for the army that stays on the general.
            if g_army - 1 < cost + BUILD_RESERVE:
                continue
            if best_cost is None or cost < best_cost:
                best_cost = cost
                best = (0, gr, gc, d, 0)
        return best

    def act(self, obs):
        structures, enemies, general = self._survey(obs)
        castles = len(structures) - (1 if general is not None else 0)
        building = obs.turn < LAST_BUILD_TURN and castles < MAX_CASTLES

        if building:
            # A stack is already sitting on a buildable cell — cash it in.
            build = self._find_build(obs, structures, enemies)
            if build is not None:
                return build
            # Otherwise walk the general's savings out to a cell we can build on.
            if general is not None:
                stage = self._stage_general(obs, structures, enemies, general)
                if stage is not None:
                    return stage

        # While saving up, keep the expander's hands off the general so its
        # army actually accumulates. Released once it holds more than any
        # castle could cost, so a general with no buildable neighbour doesn't
        # hoard forever.
        hold = general
        if not building or general is None \
                or obs.army_grid[general[0]][general[1]] > HOLD_CEILING:
            hold = None

        move = self._expander_move(obs, hold)
        if move is PASS and hold is not None:
            # Holding the general left us with nothing to do — spend it instead.
            move = self._expander_move(obs, None)
        return move

    def _expander_move(self, obs, hold):
        """The original expander policy. `hold` is a cell to leave alone."""
        best_score = -1.0
        best_move = None
        first_valid = None

        # Scan every cell on the board. The expander only ever moves armies
        # *out* of cells it already owns, so we can skip everything else.
        for r in range(obs.H):
            for c in range(obs.W):
                if obs.owner_grid[r][c] != 1:
                    continue
                if hold is not None and (r, c) == hold:
                    continue
                src_army = obs.army_grid[r][c]
                # Need at least 2 armies: one always stays behind on the source.
                if src_army <= 1:
                    continue

                # Try each of the four neighbor cells.
                for d, (dr, dc) in enumerate(DIRECTIONS):
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < obs.H and 0 <= nc < obs.W):
                        continue
                    if not _is_passable(obs.type_grid[nr][nc]):
                        continue

                    move = (0, r, c, d, 0)
                    # Remember any legal move as a fallback (used when no
                    # cell has enough army to actually capture anything).
                    if first_valid is None:
                        first_valid = move

                    dest_owner = obs.owner_grid[nr][nc]
                    dest_army = obs.army_grid[nr][nc]
                    # To capture, we need strictly more army than what's there
                    # (since one must stay on the source).
                    if src_army <= dest_army + 1:
                        continue

                    # Expansion = claiming new visible territory (vs reinforcing
                    # one of our own cells).
                    is_opp = dest_owner == 2
                    dest_type = obs.type_grid[nr][nc]
                    is_visible_neutral = (dest_owner == 0) and dest_type not in (0, 5)
                    is_expansion = is_opp or is_visible_neutral

                    # Bigger army = stronger move. Expansion is much more
                    # valuable than reinforcing; capturing the opponent
                    # specifically is worth double again.
                    score = float(src_army)
                    if is_expansion:
                        score *= 10.0
                    if is_opp:
                        score *= 2.0

                    if score > best_score:
                        best_score = score
                        best_move = move

        # Prefer the best capture; else any legal move; else pass.
        if best_move is not None:
            return best_move
        if first_valid is not None:
            return first_valid
        return PASS
