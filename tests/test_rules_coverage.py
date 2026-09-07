"""Rule-coverage tests for the JAX Generals.io engine.

Each test builds a small board by hand and computes the expected numbers by
hand in the test body, so nothing here asserts the engine against itself.
Covered rules (the ones not already exercised elsewhere in tests/):

  1. land production on the 50-turn tick (and nothing in between)
  2. tied combat against an enemy cell and against a neutral castle
  3. the global scoreboard (GameInfo and the Observation scalars)
  4. fog of war: owned cells + their 8-neighbourhood, structures in fog
  5. map generation: mountain fraction, castle count, generals present + connected
  6. truncation in GeneralsEnv: flag, auto-reset, zero reward
  7. half-move (50%) and all-in move amounts
"""
from collections import deque
from functools import lru_cache, partial

import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import pytest

from generals.core import game
from generals.core.env import GeneralsEnv
from generals.core.grid import generate_grid

UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
PASS = jnp.array([1, 0, 0, 0, 0], dtype=jnp.int32)


def move(i, j, d, split=0):
    return jnp.array([0, i, j, d, split], dtype=jnp.int32)


def both(a0, a1):
    return jnp.stack([a0, a1])


def base_grid(h, w, p0=(0, 0), p1=None):
    """Empty h x w grid with player 0's general at p0 and player 1's at p1
    (default: the opposite corner)."""
    if p1 is None:
        p1 = (h - 1, w - 1)
    grid = jnp.zeros((h, w), dtype=jnp.int32)
    return grid.at[p0].set(1).at[p1].set(2)


def give(state, player, i, j, armies):
    """Hand player `player` the cell (i, j) holding `armies` armies."""
    ownership = state.ownership.at[:, i, j].set(False).at[player, i, j].set(True)
    return state._replace(
        ownership=ownership,
        ownership_neutral=state.ownership_neutral.at[i, j].set(False),
        armies=state.armies.at[i, j].set(armies),
    )


def set_armies(state, i, j, armies):
    return state._replace(armies=state.armies.at[i, j].set(armies))


def run_passes(state, n):
    """Advance `n` turns with both players passing."""
    for _ in range(n):
        state, _ = game.step(state, both(PASS, PASS))
    return state


# ---------------------------------------------------------------------------
# 1. Land production
# ---------------------------------------------------------------------------

def test_land_production_every_50_turns_only_on_owned_plain_cells():
    """Every owned plain cell gains exactly one army when the turn counter hits a
    multiple of 50, and nothing on any other turn; neutral cells and mountains
    never grow.

    Tick convention (generals/core/game.py, `step` -> `global_update`): step()
    first increments `time` and then runs global_update on the NEW value, and
    the land tick fires when `time % 50 == 0`. So the k-th call to step() leaves
    time == k and the land bonus lands on the step that makes time 50, 100, ...
    (never on the initial state, whose time is 0 but which has not been stepped).
    Structures follow the same convention on even ticks: the comment in
    global_update explains that generals.io has a spawn frame plus a first frame
    before production starts, so the first structure increment is at time 2.
    """
    grid = base_grid(4, 4).at[2, 0].set(-2)             # mountain at (2, 0)
    state = game.create_initial_state(grid)
    state = give(state, 0, 0, 1, 5)                     # P0 plain cell, 5 armies
    state = give(state, 1, 3, 2, 2)                     # P1 plain cell, 2 armies
    state = set_armies(state, 1, 1, 3)                  # neutral plain cell holding 3 (never grows)

    for t in range(1, 101):
        state, info = game.step(state, both(PASS, PASS))
        assert int(state.time) == t
        land_ticks = t // 50                            # 1 after the step to t=50, 2 at t=100
        structure_ticks = t // 2                        # generals grow on even ticks starting at 2
        # owned plain cells: +1 per land tick, nothing else
        assert int(state.armies[0, 1]) == 5 + land_ticks, f"t={t}"
        assert int(state.armies[3, 2]) == 2 + land_ticks, f"t={t}"
        # generals: structure production plus the land tick
        assert int(state.armies[0, 0]) == 1 + structure_ticks + land_ticks, f"t={t}"
        assert int(state.armies[3, 3]) == 1 + structure_ticks + land_ticks, f"t={t}"
        # neutral cell and mountain never grow
        assert int(state.armies[1, 1]) == 3, f"t={t}"
        assert int(state.armies[2, 0]) == 0, f"t={t}"
        # ownership is untouched by production
        assert bool(state.ownership_neutral[1, 1])
        assert not bool(state.ownership[0, 1, 1]) and not bool(state.ownership[1, 1, 1])

    # exactly two land ticks happened in 100 turns: at t=50 and t=100
    assert int(state.armies[0, 1]) == 7
    assert int(state.armies[3, 2]) == 4


# ---------------------------------------------------------------------------
# 2. Tied combat
# ---------------------------------------------------------------------------

def test_tied_attack_on_enemy_cell_leaves_defender_with_zero():
    """Attacking with exactly as many armies as the defender holds is a loss for
    the attacker: the cell stays with the defender, its army drops to 0, and the
    attacker keeps the 1 army it must leave behind."""
    state = game.create_initial_state(base_grid(4, 4))
    state = set_armies(state, 0, 0, 6)                  # P0 general: 6 -> moves 5 all-in
    state = give(state, 1, 0, 1, 5)                     # P1 defends (0, 1) with 5

    state, _ = game.step(state, both(move(0, 0, RIGHT), PASS))   # time 1: no production

    assert int(state.armies[0, 0]) == 1                 # 6 - 5
    assert int(state.armies[0, 1]) == 0                 # 5 - 5
    assert bool(state.ownership[1, 0, 1])               # still P1's
    assert not bool(state.ownership[0, 0, 1])
    assert not bool(state.ownership_neutral[0, 1])
    assert int(state.winner) == -1


def test_tied_attack_on_neutral_castle_leaves_it_neutral_with_zero():
    """Same rule against a neutral castle: 40 vs 40 leaves a neutral castle
    with 0 armies, still a castle, and the attacker's cell holding 1."""
    grid = base_grid(4, 4).at[0, 2].set(40)             # neutral castle worth 40
    state = game.create_initial_state(grid)
    assert bool(state.castles[0, 2]) and int(state.armies[0, 2]) == 40
    state = give(state, 0, 0, 1, 41)                    # P0 attacks from (0, 1) with 41 -> moves 40

    state, _ = game.step(state, both(move(0, 1, RIGHT), PASS))

    assert int(state.armies[0, 1]) == 1
    assert int(state.armies[0, 2]) == 0
    assert bool(state.ownership_neutral[0, 2])
    assert bool(state.castles[0, 2])
    assert not bool(state.ownership[0, 0, 2]) and not bool(state.ownership[1, 0, 2])

    # An unowned castle at 0 does not produce; the next even tick only grows the generals.
    state = run_passes(state, 1)                        # time 2
    assert int(state.armies[0, 2]) == 0
    assert int(state.armies[0, 0]) == 2


# ---------------------------------------------------------------------------
# 3. Scoreboard
# ---------------------------------------------------------------------------

def _true_totals(state):
    """Per-player (land, army) straight from the state arrays."""
    own = np.asarray(state.ownership)
    armies = np.asarray(state.armies)
    return own.sum(axis=(1, 2)), (own * armies[None]).sum(axis=(1, 2))


def test_scoreboard_matches_true_totals_through_capture_and_production():
    """GameInfo.land / .army and the Observation scalars equal the true
    per-player totals before a capture, after it, and after the 50-turn tick.
    The scoreboard is global: a player's observation reports the opponent's
    true totals even for cells hidden by fog."""
    grid = base_grid(4, 4).at[1, 2].set(40)             # neutral castle (counts for nobody)
    state = game.create_initial_state(grid)
    state = give(state, 0, 0, 1, 10)                    # P0: general(1) + 10
    state = give(state, 1, 0, 2, 4)                     # P1: general(1) + 4 + 3
    state = give(state, 1, 3, 2, 3)

    # --- turn 1: both pass; hand totals: P0 land 2 / army 11, P1 land 3 / army 8
    state, info = game.step(state, both(PASS, PASS))
    assert info.land.tolist() == [2, 3]
    assert info.army.tolist() == [11, 8]

    # --- turn 2: P0 attacks (0,1)->(0,2) all-in: 9 vs 4, wins, 5 left on (0,2).
    # time becomes 2 (even) so both generals grow by 1.
    # P0: general 2 + (0,1) 1 + (0,2) 5 = 8 over 3 cells; P1: general 2 + 3 = 5 over 2 cells.
    state, info = game.step(state, both(move(0, 1, RIGHT), PASS))
    assert int(state.time) == 2
    assert info.land.tolist() == [3, 2]
    assert info.army.tolist() == [8, 5]

    # --- passes up to turn 50: generals +1 on every even tick (24 more: 4..50),
    # and at t=50 the land tick adds +1 to every owned cell (generals included).
    # P0: general 2+24+1 = 27, (0,1) 2, (0,2) 6 -> 35 over 3 cells
    # P1: general 27, (3,2) 4 -> 31 over 2 cells
    state = run_passes(state, 47)
    assert int(state.time) == 49
    state, info = game.step(state, both(PASS, PASS))
    assert int(state.time) == 50
    assert info.land.tolist() == [3, 2]
    assert info.army.tolist() == [35, 31]

    # the hand numbers agree with the raw arrays (sanity, not the code under test)
    land, army = _true_totals(state)
    assert land.tolist() == [3, 2] and army.tolist() == [35, 31]

    # Observation scalars: own totals and the opponent's TRUE totals.
    obs0 = game.get_observation(state, 0)
    obs1 = game.get_observation(state, 1)
    assert int(obs0.owned_land_count) == 3 and int(obs0.owned_army_count) == 35
    assert int(obs0.opponent_land_count) == 2 and int(obs0.opponent_army_count) == 31
    assert int(obs1.owned_land_count) == 2 and int(obs1.owned_army_count) == 31
    assert int(obs1.opponent_land_count) == 3 and int(obs1.opponent_army_count) == 35
    # P0 holds row 0 only, so it sees rows 0-1: P1's cells (3,2), (3,3) are in fog,
    # yet the scoreboard still reports them (the real game's scoreboard is global).
    assert not bool(obs0.opponent_cells[3, 3]) and not bool(obs0.opponent_cells[3, 2])
    assert bool(obs0.fog_cells[3, 3]) and bool(obs0.fog_cells[3, 2])
    assert int(obs0.armies[3, 3]) == 0


# ---------------------------------------------------------------------------
# 4. Fog of war
# ---------------------------------------------------------------------------

def test_fog_of_war_is_owned_cells_plus_8_neighbourhood():
    """On a 7x7 board where P0 holds the corner (0,0) and the centre (3,3), the
    visible set is exactly those cells plus their 8-neighbourhoods (clipped at
    the border): 4 + 9 = 13 cells. structures_in_fog marks exactly the fogged
    mountain and castle; an enemy army one step outside the halo is invisible
    and reads as fog."""
    grid = (base_grid(7, 7, p0=(0, 0), p1=(6, 6))
            .at[1, 1].set(-2)                           # mountain inside the corner halo
            .at[5, 5].set(-2)                           # mountain in fog
            .at[4, 4].set(40)                           # castle inside the centre halo
            .at[0, 6].set(45))                          # castle in fog
    state = game.create_initial_state(grid)
    state = give(state, 0, 3, 3, 9)                     # P0's centre cell
    state = give(state, 1, 5, 3, 7)                     # enemy army just outside the halo (rows 2..4)
    state = set_armies(state, 2, 2, 4)                  # neutral army inside the halo

    visible_cells = {(0, 0), (0, 1), (1, 0), (1, 1)} | {(i, j) for i in (2, 3, 4) for j in (2, 3, 4)}
    assert len(visible_cells) == 13
    expected_visible = np.zeros((7, 7), dtype=bool)
    for ij in visible_cells:
        expected_visible[ij] = True

    vis = np.asarray(game.get_visibility(state.ownership[0]))
    assert np.array_equal(vis, expected_visible)

    obs = game.get_observation(state, 0)
    fog = np.asarray(obs.fog_cells)
    sif = np.asarray(obs.structures_in_fog)
    # fog_cells and structures_in_fog partition the invisible cells
    assert np.array_equal(~(fog | sif), expected_visible)
    assert not np.any(fog & sif)
    # structures_in_fog: exactly the fogged mountain and the fogged castle
    expected_sif = np.zeros((7, 7), dtype=bool)
    expected_sif[5, 5] = True
    expected_sif[0, 6] = True
    assert np.array_equal(sif, expected_sif)
    # visible structures are reported in their own planes, fogged ones are not
    assert np.asarray(obs.mountains).sum() == 1 and bool(obs.mountains[1, 1])
    assert np.asarray(obs.castles).sum() == 1 and bool(obs.castles[4, 4])
    # the enemy army at (5,3) is outside the halo: unseen, zero army, fog
    assert not bool(obs.opponent_cells[5, 3])
    assert int(obs.armies[5, 3]) == 0
    assert bool(obs.fog_cells[5, 3])
    # the enemy general at (6,6) is unseen too
    assert np.asarray(obs.generals).sum() == 1 and bool(obs.generals[0, 0])
    assert not np.any(np.asarray(obs.opponent_cells))
    # what IS visible carries its true values
    assert int(obs.armies[3, 3]) == 9 and bool(obs.owned_cells[3, 3])
    assert int(obs.armies[2, 2]) == 4 and bool(obs.neutral_cells[2, 2])
    assert int(obs.armies[4, 4]) == 40 and bool(obs.neutral_cells[4, 4])
    assert np.asarray(obs.owned_cells).sum() == 2
    # every visible non-structure cell is exactly one of owned / opponent / neutral
    classes = (np.asarray(obs.owned_cells).astype(int) + np.asarray(obs.opponent_cells).astype(int)
               + np.asarray(obs.neutral_cells).astype(int))
    assert np.array_equal(classes == 1, expected_visible & ~np.asarray(state.mountains))
    assert np.all(classes[~expected_visible] == 0)


# ---------------------------------------------------------------------------
# 5. Map generation
# ---------------------------------------------------------------------------

def _bfs_distance(passable, src, dst):
    """Plain-Python BFS over 4-connected passable cells; None if unreachable."""
    h, w = passable.shape
    seen = {src}
    q = deque([(src, 0)])
    while q:
        (i, j), d = q.popleft()
        if (i, j) == dst:
            return d
        for ni, nj in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)):
            if 0 <= ni < h and 0 <= nj < w and passable[ni, nj] and (ni, nj) not in seen:
                seen.add((ni, nj))
                q.append(((ni, nj), d + 1))
    return None


# 200 boards, 15x15, generated once and shared by the two tests below.
_MAP_H, _MAP_W = 15, 15
_MAP_AREA = _MAP_H * _MAP_W
_DENSITY = (0.20, 0.30)
_CASTLES = (4, 6)
_NUM_MAPS = 200


@lru_cache(maxsize=1)
def _generated_grids():
    gen = jax.jit(jax.vmap(partial(
        generate_grid, grid_dims=(_MAP_H, _MAP_W), mountain_density_range=_DENSITY,
        num_castles_range=_CASTLES, min_generals_distance=8, castle_val_range=(40, 51))))
    return np.asarray(gen(jrandom.split(jrandom.PRNGKey(2026), _NUM_MAPS)))


def test_generated_grids_respect_density_and_have_connected_generals():
    """Over 200 boards: the mountain fraction lies in the requested band, both
    generals exist exactly once, and they are connected over open ground.

    What generate_grid promises (generals/core/grid.py): num_mountains is drawn
    uniformly from [floor(lo*area), floor(hi*area)] over the PLAYABLE area, then
    the castles are carved OUT of those mountains, so the realised mountain
    count can undershoot the band by at most num_castles_range[1]. Padding
    beyond grid_dims is mountains and is excluded from the count."""
    grids = _generated_grids()
    h, w, area = _MAP_H, _MAP_W, _MAP_AREA
    lo, hi = _DENSITY

    assert grids.shape == (_NUM_MAPS, 16, 16)           # default pad_to = max(h, w) + 1
    assert np.all(grids[:, h:, :] == -2) and np.all(grids[:, :, w:] == -2)

    for g in grids[:, :h, :w]:
        mountains = int((g == -2).sum())
        frac = mountains / area
        assert lo - _CASTLES[1] / area <= frac <= hi, frac
        # castle values in the requested range, no stray cell values
        assert np.all((g[g > 2] >= 40) & (g[g > 2] <= 50))
        assert set(np.unique(g)).issubset({-2, 0, 1, 2} | set(range(40, 51)))
        # exactly one general per player
        assert int((g == 1).sum()) == 1 and int((g == 2).sum()) == 1
        # connected over open ground (castles are walkable, mountains are not)
        a = tuple(int(x) for x in np.argwhere(g == 1)[0])
        b = tuple(int(x) for x in np.argwhere(g == 2)[0])
        assert _bfs_distance(g != -2, a, b) is not None

    # the whole band is used, not just one corner of it
    per_board = (grids[:, :h, :w] == -2).sum(axis=(1, 2))
    assert per_board.min() < per_board.max()


@pytest.mark.xfail(
    reason="generate_grid places generals on any passable cell, and castles are passable "
           "(the near-castle spawn bias even favours them), so a general can overwrite a "
           "castle: ~7% of boards on this config end up with fewer castles than "
           "num_castles_range[0]. Genuine generator bug, engine left unchanged.",
    strict=False,
)
def test_generated_grids_respect_castle_count_range():
    """Every board carries a castle count inside num_castles_range, and
    mountains + castles equals the terrain count the generator drew, which is
    inside [floor(lo*area), floor(hi*area)]. Fails today: see the xfail reason."""
    grids = _generated_grids()
    h, w, area = _MAP_H, _MAP_W, _MAP_AREA
    lo, hi = _DENSITY
    min_drawn = int(np.floor(lo * area))                # 45
    max_drawn = int(np.floor(hi * area))                # 67
    for g in grids[:, :h, :w]:
        mountains = int((g == -2).sum())
        castles = int((g > 2).sum())
        assert _CASTLES[0] <= castles <= _CASTLES[1], castles
        assert min_drawn <= mountains + castles <= max_drawn


# ---------------------------------------------------------------------------
# 6. Truncation
# ---------------------------------------------------------------------------

def test_truncation_flag_auto_reset_and_zero_reward():
    """With truncation=T the truncated flag is set exactly on step T (terminated
    stays False), both rewards are 0, and the env swaps in the next pool board."""
    T = 7
    env = GeneralsEnv(grid_dims=(6, 6), truncation=T, pool_size=3)
    pool, state = env.reset(jrandom.PRNGKey(0))
    passes = both(PASS, PASS)

    for t in range(1, T):
        ts, state = env.step(state, passes, pool)
        assert not bool(ts.truncated), f"step {t}"
        assert not bool(ts.terminated), f"step {t}"
        assert ts.reward.tolist() == [0.0, 0.0]
        assert int(state.time) == t

    ts, state = env.step(state, passes, pool)           # step T
    assert bool(ts.truncated)
    assert not bool(ts.terminated)
    assert ts.reward.tolist() == [0.0, 0.0]
    assert int(ts.info.winner) == -1
    assert int(ts.last_state.time) == T                 # the board that was truncated
    # auto-reset: the returned state is pool[0] with the pool cursor advanced
    assert int(state.time) == 0
    assert int(state.pool_idx) == 1
    first = jax.tree.map(lambda x: x[0], pool)
    assert np.array_equal(np.asarray(state.armies), np.asarray(first.armies))
    assert np.array_equal(np.asarray(state.generals), np.asarray(first.generals))
    assert np.array_equal(np.asarray(state.mountains), np.asarray(first.mountains))
    assert np.array_equal(np.asarray(state.ownership), np.asarray(first.ownership))

    # the fresh episode counts from 0 again and is not truncated at its first step
    ts, state = env.step(state, passes, pool)
    assert not bool(ts.truncated) and not bool(ts.terminated)
    assert int(state.time) == 1 and int(state.pool_idx) == 1


# ---------------------------------------------------------------------------
# 7. Half-move and all-in
# ---------------------------------------------------------------------------

def test_half_move_moves_floor_half_and_all_in_leaves_one():
    """The 50% move sends floor(n/2) and leaves ceil(n/2) behind (generals.io
    convention: the larger half stays home); the full move leaves exactly 1."""
    # odd: 7 -> move 3, keep 4
    state = set_armies(game.create_initial_state(base_grid(4, 4)), 0, 0, 7)
    state, _ = game.step(state, both(move(0, 0, RIGHT, split=1), PASS))
    assert int(state.armies[0, 0]) == 4
    assert int(state.armies[0, 1]) == 3
    assert bool(state.ownership[0, 0, 1])

    # even: 8 -> move 4, keep 4
    state = set_armies(game.create_initial_state(base_grid(4, 4)), 0, 0, 8)
    state, _ = game.step(state, both(move(0, 0, DOWN, split=1), PASS))
    assert int(state.armies[0, 0]) == 4
    assert int(state.armies[1, 0]) == 4

    # all-in: 7 -> move 6, keep 1
    state = set_armies(game.create_initial_state(base_grid(4, 4)), 0, 0, 7)
    state, _ = game.step(state, both(move(0, 0, RIGHT, split=0), PASS))
    assert int(state.armies[0, 0]) == 1
    assert int(state.armies[0, 1]) == 6

    # a cell holding 1 cannot move anything, half or full
    for split in (0, 1):
        state = game.create_initial_state(base_grid(4, 4))        # general holds 1
        state, _ = game.step(state, both(move(0, 0, RIGHT, split=split), PASS))
        assert int(state.armies[0, 0]) == 1
        assert int(state.armies[0, 1]) == 0
        assert bool(state.ownership_neutral[0, 1])

    # a half-move from 3 sends 1 (floor(3/2)) and keeps 2
    state = set_armies(game.create_initial_state(base_grid(4, 4)), 0, 0, 3)
    state, _ = game.step(state, both(move(0, 0, RIGHT, split=1), PASS))
    assert int(state.armies[0, 0]) == 2
    assert int(state.armies[0, 1]) == 1
