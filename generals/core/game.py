"""
JAX game logic for Generals.io.

This module contains the core game mechanics including state management,
action execution, and observation generation. All functions are JIT-compiled
for maximum performance.

The engine is parameterized by the number of players N (a JIT-static
dimension) and a team assignment. Free-for-all is N teams of one
(teams = arange(N), the default); team modes put several players on one
team_id (2v2 is teams = [0, 0, 1, 1]). With the defaults the game is the
classic 1v1: ownership is (2, H, W), actions are (2, 5), winner is 0 or 1.

Key functions:
    - create_initial_state: Create a new game from a grid (and optional teams)
    - step: Execute one game step with actions from all players
    - get_observation: Get a player's view with (team-shared) fog of war
"""
from functools import partial
from typing import Tuple, NamedTuple, Protocol, Any

import jax
import jax.numpy as jnp
from jax import lax

from generals.core.observation import Observation


class Game(Protocol):
    """Protocol for game objects used by the GUI."""
    agents: list[str]
    channels: Any
    grid_dims: tuple[int, int]
    general_positions: dict[str, Any]
    time: int

    def get_infos(self) -> dict[str, dict[str, Any]]:
        """Return player stats."""
        ...


class GameState(NamedTuple):
    """
    Immutable game state containing all information about the game.

    Attributes:
        armies: (H, W) array of army counts per cell.
        ownership: (N, H, W) boolean arrays, ownership[i] is player i's cells.
        ownership_neutral: (H, W) boolean mask of neutral (unowned) cells.
        generals: (H, W) boolean mask of live general positions. A captured
            general turns into a castle and leaves this mask.
        castles: (H, W) boolean mask of castle positions.
        mountains: (H, W) boolean mask of mountain positions.
        passable: (H, W) boolean mask of passable cells (not mountains).
        general_positions: (N, 2) array of [row, col] where each general started.
        teams: (N,) int32 array, teams[i] is the team id of player i. Every
            team id is a player index in the default free-for-all.
        eliminated: (N,) bool array, True once player i's general has been
            captured (their territory is gone and their actions are ignored).
        time: Scalar, current game timestep.
        winner: Scalar, -1 if game ongoing, otherwise the team id of the
            last team standing (== the player index in 1v1 / free-for-all).
        pool_idx: Scalar, index into the pre-generated state pool for auto-reset.
        tunnel_limits: Optional (H, W) int32, 0 = no tunnel. A generals.io tunnel
            tile admits at most this many armies per move (map feature added in
            2026); None (the default) disables the check at no cost.
    """

    armies: jnp.ndarray
    ownership: jnp.ndarray
    ownership_neutral: jnp.ndarray
    generals: jnp.ndarray
    castles: jnp.ndarray
    mountains: jnp.ndarray
    passable: jnp.ndarray
    general_positions: jnp.ndarray
    teams: jnp.ndarray
    eliminated: jnp.ndarray
    time: jnp.ndarray
    winner: jnp.ndarray
    pool_idx: jnp.ndarray
    tunnel_limits: Any = None      # (H, W) int32, zeros unless the map has generals.io tunnels; None in hand-built states

    @property
    def cities(self):
        """Deprecated alias — castles were renamed from cities."""
        return self.castles

    @property
    def num_players(self) -> int:
        return self.ownership.shape[-3]


class GameInfo(NamedTuple):
    """
    Game statistics returned after each step.

    Attributes:
        army: (N,) array of total army counts per player.
        land: (N,) array of total land counts per player.
        is_done: Boolean, True if game has ended.
        winner: -1 if ongoing, otherwise the winning team id.
        time: Current game timestep.
    """

    army: jnp.ndarray
    land: jnp.ndarray
    is_done: jnp.ndarray
    winner: jnp.ndarray
    time: jnp.ndarray


# Direction offsets: UP, DOWN, LEFT, RIGHT
DIRECTIONS = jnp.array([[-1, 0], [1, 0], [0, -1], [0, 1]], dtype=jnp.int32)


def create_initial_state(grid: jnp.ndarray, teams=None, num_players: int | None = None,
                         tunnel_limits: jnp.ndarray | None = None) -> GameState:
    """
    Create initial game state from a numeric grid.

    Args:
        grid: 2D array with cell values:
            - -2: Mountain (impassable)
            - 0: Empty cell
            - k in 1..N: Player (k-1)'s general
            - > N (typically 20-50): Castle with that army value
        teams: Optional (N,) array of team ids, one per player. Defaults to
            free-for-all, teams = arange(N).
        num_players: N when `teams` is not given. Defaults to 2.
        tunnel_limits: Optional (H, W) per-tile cap on the army entering the
            tile (0 = none); see GameState.

    Returns:
        GameState ready for gameplay. A player whose general is missing from
        the grid starts out eliminated.
    """
    if teams is None:
        N = 2 if num_players is None else int(num_players)
        teams = jnp.arange(N, dtype=jnp.int32)
    else:
        teams = jnp.asarray(teams, dtype=jnp.int32)
        N = int(teams.shape[0])
        if num_players is not None and int(num_players) != N:
            raise ValueError(f"num_players={num_players} does not match teams of length {N}")

    ownership = jnp.stack([grid == (i + 1) for i in range(N)])
    generals = jnp.any(ownership, axis=0)

    mountains = grid == -2
    passable = grid != -2
    castles = grid > N

    ownership_neutral = passable & ~generals

    armies = jnp.where(generals, 1, 0).astype(jnp.int32)
    armies = jnp.where(castles, grid, armies)

    general_positions = jnp.stack([
        jnp.argwhere(ownership[i], size=1, fill_value=-1)[0] for i in range(N)
    ])

    return GameState(
        armies=armies,
        ownership=ownership,
        ownership_neutral=ownership_neutral,
        generals=generals,
        castles=castles,
        mountains=mountains,
        passable=passable,
        general_positions=general_positions,
        teams=teams,
        eliminated=~jnp.any(ownership, axis=(1, 2)),
        time=jnp.int32(0),
        winner=jnp.int32(-1),
        pool_idx=jnp.int32(0),
        tunnel_limits=(jnp.zeros(grid.shape, dtype=jnp.int32) if tunnel_limits is None
                       else jnp.asarray(tunnel_limits, dtype=jnp.int32)),
    )


@jax.jit
def get_visibility(ownership: jnp.ndarray) -> jnp.ndarray:
    """Compute visibility mask (3x3 around owned cells)."""
    H, W = ownership.shape
    ownership_float = ownership.astype(jnp.float32)
    padded = jnp.pad(ownership_float, 1, mode="constant", constant_values=0)

    stacked = jnp.stack(
        [
            padded[0:H, 0:W],
            padded[0:H, 1 : W + 1],
            padded[0:H, 2 : W + 2],
            padded[1 : H + 1, 0:W],
            padded[1 : H + 1, 1 : W + 1],
            padded[1 : H + 1, 2 : W + 2],
            padded[2 : H + 2, 0:W],
            padded[2 : H + 2, 1 : W + 1],
            padded[2 : H + 2, 2 : W + 2],
        ],
        axis=0,
    )

    return jnp.max(stacked, axis=0) > 0


@partial(jax.jit, static_argnames=("spoils",))
def execute_action(state: GameState, player_idx: int, action: jnp.ndarray, spoils: bool = True) -> GameState:
    """Execute a single player's action.

    Args:
        spoils: With the default True a general capture is settled in full —
            the captured player's territory passes to the capturer (see
            eliminate_player). With False only the move itself is resolved:
            the tile changes hands and the general becomes a castle, but the
            territory, elimination and win are left unsettled. Modifiers use
            this to judge the other players' moves on the board as the move
            left it, before the spoils confiscated anyone's army.
    """
    return _execute_action_incoming(state, player_idx, action, spoils)[0]


def _execute_action_incoming(state: GameState, player_idx: int, action: jnp.ndarray,
                             spoils: bool = True) -> tuple[GameState, jnp.ndarray]:
    """execute_action that also returns the army that entered the destination (0 when nothing moved)."""
    pass_turn, si, sj, direction, split_army = action

    return lax.cond(
        pass_turn == 1,
        lambda s: (s, jnp.int32(0)),
        lambda s: _execute_move(s, player_idx, si, sj, direction, split_army, spoils),
        state,
    )


def _execute_move(state: GameState, player_idx: int, si: int, sj: int, direction: int, split_army: int,
                  spoils: bool = True) -> tuple[GameState, jnp.ndarray]:
    """Execute move logic. Returns the new state and the army that entered the destination."""
    H, W = state.armies.shape

    in_bounds = (si >= 0) & (si < H) & (sj >= 0) & (sj < W)

    di = si + DIRECTIONS[direction, 0]
    dj = sj + DIRECTIONS[direction, 1]
    dest_in_bounds = (di >= 0) & (di < H) & (dj >= 0) & (dj < W)

    owns_source = state.ownership[player_idx, si, sj]
    source_army = state.armies[si, sj]

    army_to_move = lax.cond(split_army == 1, lambda a: a // 2, lambda a: a - 1, source_army)
    army_to_move = jnp.maximum(0, jnp.minimum(army_to_move, source_army - 1))
    if state.tunnel_limits is not None:
        # generals.io tunnel: the army entering the tile is capped at the tile's limit, the rest stays
        lim = state.tunnel_limits[jnp.clip(di, 0, H - 1), jnp.clip(dj, 0, W - 1)]
        army_to_move = jnp.where(lim > 0, jnp.minimum(army_to_move, lim), army_to_move)

    # An eliminated player owns nothing, so owns_source already fails; the
    # explicit check keeps hand-built states honest too.
    valid_move = (in_bounds & dest_in_bounds & owns_source & (army_to_move > 0)
                  & state.passable[di, dj] & ~state.eliminated[player_idx])

    return lax.cond(
        valid_move,
        lambda s: (_apply_move(s, player_idx, si, sj, di, dj, army_to_move, spoils), army_to_move),
        lambda s: (s, jnp.int32(0)),
        state,
    )


def _apply_move(state: GameState, player_idx: int, si: int, sj: int, di: int, dj: int, army_to_move: int,
                spoils: bool = True) -> GameState:
    """Apply a validated move.

    Three outcomes, decided by who holds the destination:
      - Friendly (the mover or a teammate): armies pool on the destination and
        it becomes the mover's cell.
      - Enemy or neutral: an attack; the larger force keeps the difference and
        a won attack flips the cell to the mover.
      - Enemy general: a won attack is a capture. The tile keeps the attacker's
        surplus and turns into a castle; then (spoils) eliminate_player hands
        the rest of the captured player's territory to the mover with every
        army halved, rounded up.
    """
    armies = state.armies
    ownership = state.ownership
    ownership_neutral = state.ownership_neutral
    N = ownership.shape[0]
    players = jnp.arange(N)

    mover = players == player_idx                                    # (N,) one-hot
    target_owners = ownership[:, di, dj]                             # (N,)
    same_team = state.teams == state.teams[player_idx]               # (N,)
    friendly = jnp.any(target_owners & same_team)
    target_army = armies[di, dj]

    attacker_wins = army_to_move > target_army
    takes_cell = friendly | attacker_wins
    dest_army = jnp.where(friendly, target_army + army_to_move, jnp.abs(target_army - army_to_move))

    armies = armies.at[di, dj].set(dest_army).at[si, sj].add(-army_to_move)
    ownership = ownership.at[:, di, dj].set(jnp.where(takes_cell, mover, target_owners))
    ownership_neutral = ownership_neutral.at[di, dj].set(ownership_neutral[di, dj] & ~takes_cell)

    # A general tile is always owned, so a won attack on one is a capture.
    captured = attacker_wins & ~friendly & state.generals[di, dj] & jnp.any(target_owners)
    captured_idx = jnp.argmax(target_owners)
    generals = state.generals.at[di, dj].set(state.generals[di, dj] & ~captured)
    castles = state.castles.at[di, dj].set(state.castles[di, dj] | captured)

    state = state._replace(
        armies=armies,
        ownership=ownership,
        ownership_neutral=ownership_neutral,
        generals=generals,
        castles=castles,
    )
    if spoils:
        state = lax.cond(
            captured,
            lambda s: eliminate_player(s, captured_idx, player_idx),
            lambda s: s,
            state,
        )
    return state


def eliminate_player(state: GameState, captured_idx, capturer_idx) -> GameState:
    """Settle a general capture: `captured_idx` drops out of the game.

    Every cell the captured player still holds passes to `capturer_idx` with
    its army halved (rounded up, so a lone unit survives); any general the
    captured player still holds becomes a castle; the player is marked
    eliminated. If that leaves only the capturer's team alive, the winner is
    set to that team. Idempotent, so it is safe to apply to an already
    eliminated player.
    """
    N = state.ownership.shape[0]
    players = jnp.arange(N)
    cells = state.ownership[captured_idx]                            # (H, W)
    capturer = players == capturer_idx                               # (N,)

    armies = jnp.where(cells, (state.armies + 1) // 2, state.armies)
    ownership = jnp.where(cells[None], capturer[:, None, None], state.ownership)
    castles = state.castles | (state.generals & cells)
    generals = state.generals & ~cells

    eliminated = state.eliminated | (players == captured_idx)
    capturer_team = state.teams[capturer_idx]
    last_team_standing = jnp.all(eliminated | (state.teams == capturer_team))
    winner = jnp.where((state.winner < 0) & last_team_standing, capturer_team, state.winner)

    return state._replace(
        armies=armies,
        ownership=ownership,
        generals=generals,
        castles=castles,
        eliminated=eliminated,
        winner=winner,
    )


@jax.jit
def global_update(state: GameState) -> GameState:
    """Perform army increments (every 2 turns for structures, every 50 for all)."""
    time = state.time
    armies = state.armies
    owned = jnp.any(state.ownership, axis=0).astype(jnp.int32)

    increment_all = time % 50 == 0
    armies = lax.cond(
        increment_all,
        lambda a: a + owned,
        lambda a: a,
        armies,
    )

    # Generals/castles grow every 2 ticks, on EVEN ticks. generals.io has a spawn
    # frame + a first frame before production starts, so the first increment lands
    # on tick 2, not tick 1. Matching this phase is required to replay real games
    # move-for-move (verified: 0 illegal moves across scraped generals.io replays;
    # the odd-tick phase desynced the general's army by one tick and lost games early).
    increment_structures = (time % 2 == 0)
    structure_mask = (state.generals | state.castles).astype(jnp.int32)
    armies = lax.cond(
        increment_structures,
        lambda a: a + structure_mask * owned,
        lambda a: a,
        armies,
    )

    return state._replace(armies=armies)


def _official_move_order(state: GameState, actions: jnp.ndarray) -> jnp.ndarray:
    """generals.io's current move order (``MoveResolver.determineMoveOrder`` in the client bundle,
    2025): moves are sorted defensive first (destination held by the mover's team), then
    moves that attack a general last, then LARGER army first, then player order (reversed
    on odd turns); a move whose
    source another pending move is entering (a chased piece) waits until that chaser has
    resolved, unless the two moves are a head-on swap. Passes resolve last. Returns the (N,)
    order of player indices."""
    N = actions.shape[0]
    H, W = state.armies.shape
    idx = jnp.arange(N)
    passes = actions[:, 0] != 0
    si, sj, direction = actions[:, 1], actions[:, 2], actions[:, 3]
    di = si + DIRECTIONS[direction, 0]
    dj = sj + DIRECTIONS[direction, 1]
    csi, csj = jnp.clip(si, 0, H - 1), jnp.clip(sj, 0, W - 1)
    cdi, cdj = jnp.clip(di, 0, H - 1), jnp.clip(dj, 0, W - 1)
    dest_owners = state.ownership[:, cdi, cdj]                          # (owner, mover)
    same_team = state.teams[:, None] == state.teams[None, :]
    defensive = jnp.any(dest_owners & same_team, axis=0) & ~passes
    general_attack = state.generals[cdi, cdj] & ~defensive & ~passes
    army = jnp.where(passes, -1, state.armies[csi, csj])
    # sort rank: defensive desc, general_attack asc, army desc, index asc; passes last
    dj_, di_ = defensive[:, None], defensive[None, :]
    gj_, gi_ = general_attack[:, None], general_attack[None, :]
    aj_, ai_ = army[:, None], army[None, :]
    pj_, pi_ = passes[:, None], passes[None, :]
    # equal armies: the site takes the moves in player order, reversed on odd turns
    by_index = jnp.where(state.time % 2 == 0, idx[:, None] < idx[None, :], idx[:, None] > idx[None, :])
    ahead = ((~pj_ & pi_) | ((pj_ == pi_) & ((dj_ & ~di_) | ((dj_ == di_) & ((~gj_ & gi_) | ((gj_ == gi_) &
             ((aj_ > ai_) | ((aj_ == ai_) & by_index))))))))
    rank = jnp.sum(ahead, axis=0)                                       # sort position of each player
    # dependency: j's move enters i's source and is not the head-on partner of i
    enters = (di[:, None] == si[None, :]) & (dj[:, None] == sj[None, :]) & ~passes[:, None] & ~passes[None, :]
    head_on = enters & enters.T
    dep = enters & ~head_on & ~jnp.eye(N, dtype=bool)                    # dep[j, i]: i waits for j
    order = jnp.zeros((N,), dtype=jnp.int32)
    queued = jnp.zeros((N,), dtype=bool)
    for k in range(N):
        blocked = jnp.any(dep & ~queued[:, None], axis=0)               # some unqueued move enters my source
        big = N + 1
        key_free = jnp.where(~queued & ~blocked, rank, big)
        key_any = jnp.where(~queued, rank, big)
        pick = jnp.where(jnp.min(key_free) < big, jnp.argmin(key_free), jnp.argmin(key_any))
        order = order.at[k].set(pick)
        queued = queued.at[pick].set(True)
    return order


def _determine_move_order(state: GameState, actions: jnp.ndarray,
                          legacy_move_priority: bool = False,
                          official_move_priority: bool = False) -> jnp.ndarray:
    """Order in which this turn's moves resolve: an (N,) array of player indices.

    Priority is chasing > reinforcing > SMALLER army, ties by player index,
    passes last. Chasing: the move lands on the source of another player's
    move. Reinforcing: the move lands on a cell the mover's team holds.
    Smaller army first: on a contested cell the bigger force resolves last and
    ends up holding it (larger-first let the smaller force snipe a neutral
    castle the bigger one had just paid for), and a deathtouch head-on clash
    goes to the attacker, keeping the endgame a forced finish.

    For two players this is exactly the old first-mover rule; it is computed
    with pairwise comparisons rather than a sort, so it costs a few (N, N)
    boolean ops.

    legacy_move_priority=True selects the rule generals.io used when the public
    replay archive was recorded and this engine used before April 2025
    (generals-bots commit e5676c3 introduced the rule above):
    priority simply alternates every tick, independent of the moves. Player 0
    resolves first on even ticks and last on odd ticks (for N players the
    index order is reversed on odd ticks). It exists so archived replays
    recorded under the old rule can be reproduced move-for-move; nothing in
    the environment turns it on by default.
    """
    N = actions.shape[0]
    H, W = state.armies.shape
    idx = jnp.arange(N)

    if legacy_move_priority:
        return jnp.where(state.time % 2 == 0, idx, idx[::-1])
    if official_move_priority:
        return _official_move_order(state, actions)

    passes = actions[:, 0] != 0
    si, sj, direction = actions[:, 1], actions[:, 2], actions[:, 3]
    di = si + DIRECTIONS[direction, 0]
    dj = sj + DIRECTIONS[direction, 1]

    # chasing[i]: i's destination is the source of some other player's move
    onto_source = (di[:, None] == si[None, :]) & (dj[:, None] == sj[None, :])
    chasing = jnp.any(onto_source & ~passes[None, :] & ~jnp.eye(N, dtype=bool), axis=1)

    # reinforcing[i]: i's destination is held by i's team. Out-of-bounds
    # destinations are clipped; such a move is invalid and never executes, so
    # its slot in the order is irrelevant.
    ci, cj = jnp.clip(di, 0, H - 1), jnp.clip(dj, 0, W - 1)
    dest_owners = state.ownership[:, ci, cj]                          # (owner, mover)
    same_team = state.teams[:, None] == state.teams[None, :]          # (owner, mover)
    reinforcing = jnp.any(dest_owners & same_team, axis=0)

    army = state.armies[jnp.clip(si, 0, H - 1), jnp.clip(sj, 0, W - 1)]

    c = chasing & ~passes
    r = reinforcing & ~passes
    a = jnp.where(passes, jnp.iinfo(jnp.int32).max, army)

    # ahead[j, i]: j resolves before i (lexicographic: c desc, r desc, a asc, index asc)
    cj_, ci_ = c[:, None], c[None, :]
    rj_, ri_ = r[:, None], r[None, :]
    aj_, ai_ = a[:, None], a[None, :]
    by_index = idx[:, None] < idx[None, :]
    ahead = (cj_ & ~ci_) | ((cj_ == ci_) & ((rj_ & ~ri_) | ((rj_ == ri_) & ((aj_ < ai_) | ((aj_ == ai_) & by_index)))))

    rank = jnp.sum(ahead, axis=0)                                     # players ahead of each i
    return jnp.argmax(rank[None, :] == idx[:, None], axis=1)          # slot k -> player with rank k


# --------------------------------------------------------------------------- #
# General trade (generals.io, replay format 16 and later)
# --------------------------------------------------------------------------- #
def _move_geometry(state: GameState, player_idx, action):
    """(valid, di, dj, si, sj, army_to_move, reserve) of one action on `state`,
    with the validity test of _execute_move. reserve is what stays on the source."""
    pass_turn, si, sj, direction, split_army = action
    H, W = state.armies.shape
    in_bounds = (si >= 0) & (si < H) & (sj >= 0) & (sj < W)
    di = si + DIRECTIONS[direction, 0]
    dj = sj + DIRECTIONS[direction, 1]
    dest_in_bounds = (di >= 0) & (di < H) & (dj >= 0) & (dj < W)
    csi, csj = jnp.clip(si, 0, H - 1), jnp.clip(sj, 0, W - 1)
    cdi, cdj = jnp.clip(di, 0, H - 1), jnp.clip(dj, 0, W - 1)
    owns_source = state.ownership[player_idx, csi, csj]
    source_army = state.armies[csi, csj]
    army_to_move = jnp.where(split_army == 1, source_army // 2, source_army - 1)
    army_to_move = jnp.maximum(0, jnp.minimum(army_to_move, source_army - 1))
    valid = ((pass_turn == 0) & in_bounds & dest_in_bounds & owns_source & (army_to_move > 0)
             & state.passable[cdi, cdj] & ~state.eliminated[player_idx])
    return valid, cdi, cdj, csi, csj, army_to_move, source_army - army_to_move


def _general_cell(state: GameState, player_idx):
    """(row, col) of player_idx's live general (the first cell of the mask they hold)."""
    mask = state.generals & state.ownership[player_idx]
    flat = jnp.argmax(mask.reshape(-1))
    W = state.armies.shape[1]
    return flat // W, flat % W


def _is_general_trade(state: GameState, actions: jnp.ndarray, e, t) -> jnp.ndarray:
    """True iff, on `state`, player e's move captures t's general AND t's move
    captures e's general (generals.io's ``getMutualGeneralSwapMoveIndex``).
    Each attack is judged against the garrison the defender's own move leaves
    behind when that move departs from the general (``wouldAttackCaptureGeneral``)."""
    ve, dei, dej, sei, sej, me, re_ = _move_geometry(state, e, actions[e])
    vt, dti, dtj, sti, stj, mt, rt = _move_geometry(state, t, actions[t])
    gei, gej = _general_cell(state, e)
    gti, gtj = _general_cell(state, t)
    hits = (dei == gti) & (dej == gtj) & (dti == gei) & (dtj == gej)
    enemies = state.teams[e] != state.teams[t]
    # garrison of t's general when e's army lands: t's own reserve if t moves off it
    t_from_general = (sti == gti) & (stj == gtj)
    e_from_general = (sei == gei) & (sej == gej)
    garrison_t = jnp.where(t_from_general, rt, state.armies[gti, gtj])
    garrison_e = jnp.where(e_from_general, re_, state.armies[gei, gej])
    return ve & vt & hits & enemies & (me > garrison_t) & (mt > garrison_e)


def _execute_general_trade(state: GameState, actions: jnp.ndarray, e, t) -> GameState:
    """generals.io's ``executeMutualGeneralSwap``: both captures happen at once.
    Each player's attacking army lands on the other's general with its
    surplus, every other cell of each player passes to the other with its army
    halved (rounded up), nobody is eliminated, and the two generals change
    hands: e's general is now where t's was, and vice versa."""
    _, _, _, sei, sej, me, re_ = _move_geometry(state, e, actions[e])
    _, _, _, sti, stj, mt, rt = _move_geometry(state, t, actions[t])
    gei, gej = _general_cell(state, e)
    gti, gtj = _general_cell(state, t)
    t_from_general = (sti == gti) & (stj == gtj)
    e_from_general = (sei == gei) & (sej == gej)
    garrison_t = jnp.where(t_from_general, rt, state.armies[gti, gtj])
    garrison_e = jnp.where(e_from_general, re_, state.armies[gei, gej])
    surplus_e = jnp.maximum(0, me - garrison_t)          # lands on t's general, now e's
    surplus_t = jnp.maximum(0, mt - garrison_e)          # lands on e's general, now t's

    # armies left on the two sources once the attacking stacks have departed
    armies = state.armies.at[sei, sej].set(re_)
    armies = armies.at[sti, stj].set(rt)
    N = state.ownership.shape[0]
    players = jnp.arange(N)
    own_e, own_t = state.ownership[e], state.ownership[t]
    gen_cells = jnp.zeros_like(own_e).at[gei, gej].set(True).at[gti, gtj].set(True)
    swap = (own_e | own_t) & ~gen_cells
    armies = jnp.where(swap, (armies + 1) // 2, armies)          # Math.round(0.5 * army)
    armies = armies.at[gti, gtj].set(surplus_e).at[gei, gej].set(surplus_t)

    to_t = (own_e & ~gen_cells).at[gei, gej].set(True)    # e's land and e's old general -> t
    to_e = (own_t & ~gen_cells).at[gti, gtj].set(True)    # t's land and t's old general -> e
    is_e, is_t = (players == e)[:, None, None], (players == t)[:, None, None]
    ownership = jnp.where(to_t[None], is_t, jnp.where(to_e[None], is_e, state.ownership))

    gp = state.general_positions
    gp = gp.at[e].set(jnp.array([gti, gtj], dtype=gp.dtype)).at[t].set(jnp.array([gei, gej], dtype=gp.dtype))
    return state._replace(armies=armies, ownership=ownership, general_positions=gp)


def _apply_general_trades(state: GameState, actions: jnp.ndarray) -> tuple[GameState, jnp.ndarray]:
    """Resolve every general trade this turn before the ordinary moves.
    Returns the state and an (N,) mask of the players whose moves were consumed."""
    N = actions.shape[0]
    consumed = jnp.zeros((N,), dtype=bool)
    for e in range(N):
        for t in range(e + 1, N):
            trade = _is_general_trade(state, actions, e, t) & ~consumed[e] & ~consumed[t]
            state = lax.cond(trade, lambda s: _execute_general_trade(s, actions, e, t), lambda s: s, state)
            consumed = consumed.at[e].set(consumed[e] | trade).at[t].set(consumed[t] | trade)
    return state, consumed



def _refund_tunnel_overflow(state: GameState, actions: jnp.ndarray, order: jnp.ndarray,
                            incoming: jnp.ndarray) -> GameState:
    """generals.io ``Map.refundTunnelOverflow``: after all moves of a tick, a tunnel tile holding more
    than its limit hands the excess back to the movers that entered it this tick, in move order, each
    refunded at most what it brought in."""
    H, W = state.armies.shape
    lim = state.tunnel_limits
    remaining = jnp.where(lim > 0, jnp.maximum(0, state.armies - lim), 0)
    armies = state.armies
    for k in range(order.shape[0]):
        p = order[k]
        a = actions[p]
        si, sj = jnp.clip(a[1], 0, H - 1), jnp.clip(a[2], 0, W - 1)
        di = jnp.clip(si + DIRECTIONS[a[3], 0], 0, H - 1)
        dj = jnp.clip(sj + DIRECTIONS[a[3], 1], 0, W - 1)
        o = jnp.minimum(remaining[di, dj], incoming[p])                     # 0 for passes and dropped moves
        armies = armies.at[si, sj].add(o).at[di, dj].add(-o)
        remaining = remaining.at[di, dj].add(-o)
    return state._replace(armies=armies)


@partial(jax.jit, static_argnames=("legacy_move_priority", "general_trade", "official_move_priority"))
def step(state: GameState, actions: jnp.ndarray,
         legacy_move_priority: bool = False,
         general_trade: bool = False,
         official_move_priority: bool = False) -> tuple[GameState, GameInfo]:
    """Execute one game step with actions from all players.

    Args:
        state: Current game state.
        actions: (N, 5) array, one [pass, row, col, direction, split] per player.
        legacy_move_priority: Resolve moves in the old alternating order
            instead of the current chasing > reinforcing > smaller-army rule
            (see _determine_move_order). Default False; the game is unchanged
            unless it is passed explicitly.
        general_trade: generals.io's rule (replay format 16, 2025) for two
            players who capture each other's general on the same turn: both
            captures happen, nobody is eliminated, and the generals change
            hands (see _execute_general_trade). Default False, in which case
            the first capture in move order settles the turn as before.

    Moves resolve one after another in _determine_move_order's order, each on
    the board the previous one left — a capture confiscates the captured
    player's territory immediately, so their later move (and every move after
    the game is decided) finds no army to command.
    """
    N = actions.shape[0]
    if N != state.ownership.shape[0]:
        raise ValueError(f"got actions for {N} players but the state has {state.ownership.shape[0]}")
    done_before = state.winner >= 0

    if general_trade:
        state, consumed = _apply_general_trades(state, actions)
        pass_action = jnp.array([1, 0, 0, 0, 0], dtype=actions.dtype)
        actions = jnp.where(consumed[:, None], pass_action[None, :], actions)

    order = _determine_move_order(state, actions, legacy_move_priority, official_move_priority)
    incoming = jnp.zeros((N,), dtype=jnp.int32)
    for k in range(N):
        player = order[k]
        state, inc = _execute_action_incoming(state, player, actions[player])
        incoming = incoming.at[player].set(inc)
    if state.tunnel_limits is not None:
        state = _refund_tunnel_overflow(state, actions, order, incoming)

    state = lax.cond(done_before, lambda s: s, lambda s: s._replace(time=s.time + 1), state)

    state = lax.cond(
        state.winner >= 0,
        lambda s: s,
        lambda s: global_update(s),
        state,
    )

    return state, get_info(state)


@jax.jit
def get_info(state: GameState) -> GameInfo:
    """Compute game statistics."""
    armies = state.armies
    ownership = state.ownership

    return GameInfo(
        army=jnp.sum(armies[None] * ownership, axis=(1, 2)),
        land=jnp.sum(ownership, axis=(1, 2)),
        is_done=state.winner >= 0,
        winner=state.winner,
        time=state.time,
    )


def team_cells(state: GameState, player_idx) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """(own, allied, enemy) cell masks for `player_idx`. Allied excludes self;
    in 1v1 / free-for-all it is empty and enemy is everyone else's land."""
    same_team = state.teams == state.teams[player_idx]
    own = state.ownership[player_idx]
    team = jnp.any(state.ownership & same_team[:, None, None], axis=0)
    enemy = jnp.any(state.ownership & ~same_team[:, None, None], axis=0)
    return own, team & ~own, enemy


def _observe(state: GameState, player_idx, fog: bool) -> Observation:
    N = state.ownership.shape[0]
    players = jnp.arange(N)
    same_team = state.teams == state.teams[player_idx]
    teammate = same_team & (players != player_idx)

    own_cells, allied_cells, enemy_cells = team_cells(state, player_idx)
    if fog:
        # Sight is shared within a team: 3x3 around any cell the team holds.
        visible = get_visibility(own_cells | allied_cells)
    else:
        visible = jnp.ones_like(own_cells)
    invisible = ~visible

    info = get_info(state)
    structures = state.mountains | state.castles

    return Observation(
        armies=state.armies * visible,
        generals=state.generals * visible,
        castles=state.castles * visible,
        mountains=state.mountains * visible,
        neutral_cells=state.ownership_neutral * visible,
        owned_cells=own_cells * visible,
        opponent_cells=enemy_cells * visible,
        fog_cells=invisible & ~structures,
        structures_in_fog=invisible & structures,
        owned_land_count=info.land[player_idx],
        owned_army_count=info.army[player_idx],
        opponent_land_count=jnp.sum(jnp.where(same_team, 0, info.land)),
        opponent_army_count=jnp.sum(jnp.where(same_team, 0, info.army)),
        timestep=state.time,
        allied_cells=allied_cells * visible,
        allied_land_count=jnp.sum(jnp.where(teammate, info.land, 0)),
        allied_army_count=jnp.sum(jnp.where(teammate, info.army, 0)),
    )


@jax.jit
def get_observation(state: GameState, player_idx: int) -> Observation:
    """Get player observation with fog of war applied.

    Fog is team-shared: the player sees the 3x3 neighbourhood of every cell
    their team holds. `opponent_*` covers every enemy team together and
    `allied_*` the teammates (empty in 1v1 / free-for-all).
    """
    return _observe(state, player_idx, fog=True)


@jax.jit
def get_full_observation(state: GameState, player_idx: int) -> Observation:
    """Get a player's observation with NO fog of war — everything is visible.

    Used for perfect-information variants of the competition. fog_cells and
    structures_in_fog masks are all-zeros; every other field reflects the
    true game state.
    """
    return _observe(state, player_idx, fog=False)


@jax.jit
def batch_step(states: GameState, actions: jnp.ndarray) -> Tuple[GameState, GameInfo]:
    """Vectorized step for multiple environments."""
    return jax.vmap(step)(states, actions)
