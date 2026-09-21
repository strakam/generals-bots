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
    - surrender / neutralize: generals.io's server-side events (a player
      leaves; their land is handed on), applied to the state between steps
    - get_observation: Get a player's view with (team-shared) fog of war

The rules are generals.io's (2026), verified tile for tile against the
site's own engine on ranked 1v1, free-for-all and 2v2 replays
(paper/validation).
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
        eliminated: (N,) bool array, True once player i is out of the game:
            their general was captured (their territory is gone) or they
            surrendered (see surrender: the territory stays). Their actions
            are ignored either way.
        time: Scalar, current game timestep.
        winner: Scalar, -1 if game ongoing, otherwise the team id of the
            last team standing (== the player index in 1v1 / free-for-all).
        pool_idx: Scalar, index into the pre-generated state pool for auto-reset.
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


def create_initial_state(grid: jnp.ndarray, teams=None, num_players: int | None = None) -> GameState:
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
    pass_turn, si, sj, direction, split_army = action

    return lax.cond(
        pass_turn == 1,
        lambda s: s,
        lambda s: _execute_move(s, player_idx, si, sj, direction, split_army, spoils),
        state,
    )


def _execute_move(state: GameState, player_idx: int, si: int, sj: int, direction: int, split_army: int,
                  spoils: bool = True) -> GameState:
    """Execute move logic."""
    H, W = state.armies.shape

    in_bounds = (si >= 0) & (si < H) & (sj >= 0) & (sj < W)

    di = si + DIRECTIONS[direction, 0]
    dj = sj + DIRECTIONS[direction, 1]
    dest_in_bounds = (di >= 0) & (di < H) & (dj >= 0) & (dj < W)

    owns_source = state.ownership[player_idx, si, sj]
    source_army = state.armies[si, sj]

    army_to_move = lax.cond(split_army == 1, lambda a: a // 2, lambda a: a - 1, source_army)
    army_to_move = jnp.maximum(0, jnp.minimum(army_to_move, source_army - 1))

    # generals.io's checkAttackValid: a source with 1 army may still move onto
    # a TEAMMATE's tile ("1 !== armyAt(t) || teams[e] === teams[tileAt(n)] &&
    # tileAt(n) !== e"). No army moves; _apply_move hands the tile over when it
    # is a teammate's (not their general) and changes nothing otherwise, so
    # the destination need not be inspected here.
    can_move = (army_to_move > 0) | (source_army == 1)

    # A surrendered player still owns land but may not move it; for a
    # captured player owns_source already fails.
    valid_move = (in_bounds & dest_in_bounds & owns_source & can_move
                  & state.passable[di, dj] & ~state.eliminated[player_idx])

    return lax.cond(
        valid_move,
        lambda s: _apply_move(s, player_idx, si, sj, di, dj, army_to_move, spoils),
        lambda s: s,
        state,
    )


def _apply_move(state: GameState, player_idx: int, si: int, sj: int, di: int, dj: int, army_to_move: int,
                spoils: bool = True) -> GameState:
    """Apply a validated move (generals.io's Map.attack).

    Three outcomes, decided by who holds the destination:
      - Friendly (the mover or a teammate): armies pool on the destination and
        it becomes the mover's cell — except a teammate's general, which keeps
        its owner ("a !== s && generals[a] !== t && setTile(t, s)").
      - Enemy or neutral: an attack; the larger force keeps the difference and
        a won attack flips the cell to the mover.
      - Enemy general: a won attack is a capture. The tile keeps the attacker's
        surplus and turns into a castle; then (spoils) eliminate_player hands
        the rest of the captured player's territory to the mover with every
        army halved, rounded up.
    A 1-army source moves nothing (army_to_move == 0): a teammate's ordinary
    tile changes hands and every other destination is left exactly as it was.
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
    ally_general = friendly & ~target_owners[player_idx] & state.generals[di, dj]
    target_army = armies[di, dj]

    attacker_wins = army_to_move > target_army
    takes_cell = jnp.where(friendly, ~ally_general, attacker_wins)
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
def surrender(state: GameState, player_idx) -> GameState:
    """generals.io's ``killPlayer`` (a player surrenders or is kicked for
    inactivity; the first AFK event of a replay): the player is dead, so
    their moves are rejected from now on, but every tile stays theirs and
    keeps receiving income, and their general can still be captured (the
    capturer then takes the land, halved, as usual). The game ends when one
    team is left alive: the next step sets the winner, completing that tick
    like the tick of a final capture (moves run, time advances, no income).
    Idempotent. Pure; apply it to the state before the tick's step.
    """
    N = state.ownership.shape[0]
    return state._replace(eliminated=state.eliminated | (jnp.arange(N) == player_idx))


@jax.jit
def neutralize(state: GameState, player_idx) -> GameState:
    """generals.io's ``tryNeutralizePlayer`` (the second AFK event of a
    replay, 50 turns after the kill, or 1 turn after it in team games for a
    player with few tiles): if the player's general tile is still theirs,
    every tile they hold passes to their first living teammate
    (``getNextLivingTeammateIndex``) or, without one, becomes neutral, armies
    unchanged (``replaceAll(p, q)``), and the general becomes a castle. If
    the general has fallen already, nothing happens. Pure; apply it to the
    state before the tick's step.
    """
    N = state.ownership.shape[0]
    players = jnp.arange(N)
    gi, gj = state.general_positions[player_idx]
    still_mine = state.ownership[player_idx, gi, gj] & state.generals[gi, gj]
    cells = state.ownership[player_idx] & still_mine
    mates = (state.teams == state.teams[player_idx]) & ~state.eliminated & (players != player_idx)
    has_mate = jnp.any(mates)
    heir = (players == jnp.argmax(mates)) & has_mate                   # (N,) one-hot, or nobody
    return state._replace(
        ownership=jnp.where(cells[None], heir[:, None, None], state.ownership),
        ownership_neutral=state.ownership_neutral | (cells & ~has_mate),
        castles=state.castles | (state.generals & cells),
        generals=state.generals & ~cells,
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


def _determine_move_order(state: GameState, actions: jnp.ndarray) -> jnp.ndarray:
    """Order in which this turn's moves resolve: an (N,) array of player indices.

    This is generals.io's current rule (``MoveResolver.determineMoveOrder`` in
    the client, 2025), so that the engine replicates the website game: moves
    are sorted defensive first (destination held by the mover's team), then
    moves onto a general tile last (any general, a friendly merge included),
    then LARGER army first, then player order (reversed on odd turns); a move
    whose source another pending move is entering (a chased piece) waits until
    that chaser has resolved, unless the two moves are a head-on swap. Passes
    resolve last. Verified tile for tile against the site's own engine on
    10,000 ranked 1v1, 1,000 FFA and 1,000 2v2 replays (paper/validation).
    """
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
    # isGeneralAttack: the destination is a general tile, a friendly merge onto one included
    general_attack = state.generals[cdi, cdj] & ~passes
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
    can_move = (army_to_move > 0) | (source_army == 1)
    valid = ((pass_turn == 0) & in_bounds & dest_in_bounds & owns_source & can_move
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


def _moves_with_general_trades(state: GameState, actions: jnp.ndarray, order: jnp.ndarray) -> GameState:
    """Execute the turn's moves in `order`, resolving general trades where
    generals.io's ``Game.update`` does: when the walk reaches a move, it looks
    for a LATER move of the target player onto the mover's general such that
    both attacks capture on the board as it stands (``getMutualGeneralSwapMoveIndex``);
    if there is one the trade executes at this position and the later move is
    marked used. Everything else in the turn runs on the board the trade left
    (a third player's move between the two positions included).
    """
    N = actions.shape[0]
    consumed = jnp.zeros((N,), dtype=bool)
    for k in range(N):
        player = order[k]
        for m in range(k + 1, N):
            other = order[m]
            trade = _is_general_trade(state, actions, player, other) & ~consumed[player] & ~consumed[other]
            state = lax.cond(trade, lambda s: _execute_general_trade(s, actions, player, other), lambda s: s, state)
            consumed = consumed.at[player].set(consumed[player] | trade).at[other].set(consumed[other] | trade)
        state = lax.cond(consumed[player], lambda s: s, lambda s: execute_action(s, player, actions[player]), state)
    return state


def _last_team_standing(state: GameState) -> jnp.ndarray:
    """generals.io's ``isOver``: the team id of the only team with a living
    player, or -1 while two teams are alive (or nobody is)."""
    alive = ~state.eliminated
    team = state.teams[jnp.argmax(alive)]
    over = jnp.any(alive) & jnp.all(~alive | (state.teams == team))
    return jnp.where(over, team, jnp.int32(-1))


@partial(jax.jit, static_argnames=("general_trade",))
def step(state: GameState, actions: jnp.ndarray,
         general_trade: bool = False) -> tuple[GameState, GameInfo]:
    """Execute one game step with actions from all players.

    Args:
        state: Current game state.
        actions: (N, 5) array, one [pass, row, col, direction, split] per player.
        general_trade: generals.io's rule (replay format 16, 2025) for two
            players who capture each other's general on the same turn: both
            captures happen, nobody is eliminated, and the generals change
            hands (see _execute_general_trade). Default False, in which case
            the first capture in move order settles the turn as before.

    Moves resolve one after another in _determine_move_order's order, each on
    the board the previous one left — a capture confiscates the captured
    player's territory immediately, so their later move (and every move after
    the game is decided) finds no army to command.

    The tick that decides the game (by capture or, via surrender, by the last
    opposing player leaving) completes: its moves run and time advances, but
    the income is not paid. Later ticks leave time alone.
    """
    N = actions.shape[0]
    if N != state.ownership.shape[0]:
        raise ValueError(f"got actions for {N} players but the state has {state.ownership.shape[0]}")
    done_before = state.winner >= 0

    order = _determine_move_order(state, actions)
    if general_trade:
        state = _moves_with_general_trades(state, actions, order)
    else:
        for k in range(N):
            player = order[k]
            state = execute_action(state, player, actions[player])

    state = lax.cond(done_before, lambda s: s, lambda s: s._replace(time=s.time + 1), state)

    # A surrender may have left one team standing (generals.io's isOver runs
    # in killPlayer): the tick completes like the tick of a final capture.
    state = state._replace(winner=jnp.where(state.winner >= 0, state.winner, _last_team_standing(state)))

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
