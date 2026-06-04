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
    obs.type_grid[r][c]     0=fog, 1=plain, 2=mountain, 3=city, 4=general, 5=structure-in-fog
    obs.owner_grid[r][c]    0=neutral/unknown, 1=me, 2=opp  (perspective-relative)
    obs.army_grid[r][c]     army count, 0 in fog or empty

`act` must return a 5-tuple `(pass, row, col, direction, split)`:

    pass:       1 to skip the turn, 0 to move
    row, col:   source cell (must be owned by you and have army > 1)
    direction:  0=up, 1=down, 2=left, 3=right
    split:      0=move all-but-one armies, 1=move half (floor division)

Invalid moves are silently treated as a pass by the engine.
"""

# A no-op action — used when no valid move exists or as a safe default.
PASS = (1, 0, 0, 0, 0)

# (dr, dc) offsets for direction codes 0..3
DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _is_passable(t):
    # Mountains (2) and fogged-structures (5) are impassable; everything else
    # can be entered (including fog — you just don't know what's there).
    return t != 2 and t != 5


class Agent:
    """Expander strategy.

    Each turn pick the move that maximizes
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

    def act(self, obs):
        best_score = -1.0
        best_move = None
        first_valid = None

        # Scan every cell on the board. The expander only ever moves armies
        # *out* of cells it already owns, so we can skip everything else.
        for r in range(obs.H):
            for c in range(obs.W):
                if obs.owner_grid[r][c] != 1:
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
