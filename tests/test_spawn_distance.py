"""Generals spawn at a real WALKING distance, not a straight-line one.

The generator places terrain first, runs one BFS dilation from general A, and
draws general B from the cells at least `min_generals_distance` steps away. So
two generals with a ridge between them count as far apart even when the straight
line is short — and, unlike the old Manhattan proxy, a board is never rejected
for a short straight line that is actually a long walk.

These tests check the distance field against an independent BFS, then measure
the property that matters on real competition boards: the walking distance
between generals on the FINAL grid the evaluator plays.
"""
from collections import deque

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from generals.core.env import GeneralsEnv
from generals.core.grid import bfs_distance_field, generate_grid
from generals.modifiers import build_castles as bc

COMPETITION_MIN_DISTANCE = 17


# --------------------------------------------------------------- helpers


def ref_bfs(passable, start):
    """Plain-Python BFS — the independent oracle for the jitted dilation."""
    h, w = passable.shape
    dist = {start: 0}
    q = deque([start])
    while q:
        r, c = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and passable[nr, nc] and (nr, nc) not in dist:
                dist[(nr, nc)] = dist[(r, c)] + 1
                q.append((nr, nc))
    return dist


def generals_of(grid):
    g = np.asarray(grid)
    (ar, ac) = np.argwhere(g == 1)[0]
    (br, bc) = np.argwhere(g == 2)[0]
    return (int(ar), int(ac)), (int(br), int(bc))


def walk_distance(grid):
    """Walking distance between the two generals, by the game's own rule.

    game.create_initial_state uses `passable = grid != -2` — only mountains
    block. Castles are walkable (capturable), and under the competition ruleset
    they are stripped to plain ground anyway, so this one measure is what the
    match is played on both before and after stripping.
    """
    g = np.asarray(grid)
    a, b = generals_of(g)
    return ref_bfs(g != -2, a).get(b, None)


def competition_board(env, seed, min_distance):
    """One board the way eval/run_match.py builds it: exact rectangle, no pad."""
    kd, kg = jax.random.split(jax.random.PRNGKey(seed))
    lo, hi = env.min_grid_size, env.max_grid_size
    h = int(jax.random.randint(kd, (), lo, hi + 1))
    w = int(jax.random.randint(jax.random.fold_in(kd, 1), (), lo, hi + 1))
    return generate_grid(
        kg, grid_dims=(h, w),
        mountain_density_range=env.mountain_density_range,
        num_castles_range=env.num_castles_range,
        min_generals_distance=min_distance,
        castle_val_range=env.castle_val_range,
    )[:h, :w]


# ------------------------------------------------------- the distance field


@pytest.mark.parametrize("seed", range(6))
def test_distance_field_matches_a_plain_bfs(seed):
    rng = np.random.default_rng(seed)
    passable = rng.random((12, 15)) > 0.3
    start = tuple(np.argwhere(passable)[0])
    passable[start] = True

    got = np.asarray(bfs_distance_field(jnp.array(passable), start))
    want = ref_bfs(passable, start)
    h, w = passable.shape

    for r in range(h):
        for c in range(w):
            if (r, c) in want:
                assert got[r, c] == want[(r, c)], f"cell {(r, c)}"
            else:
                assert got[r, c] == h * w, f"cell {(r, c)} should be unreachable"


def test_distance_field_walks_around_a_wall():
    """A ridge with one gap: the straight line is short, the walk is not."""
    passable = np.ones((9, 9), dtype=bool)
    passable[:, 4] = False        # full-height wall...
    passable[8, 4] = True         # ...with a single gap at the bottom
    got = np.asarray(bfs_distance_field(jnp.array(passable), (0, 0)))

    # (0,8) is 8 steps away in a straight line but must go down and around
    assert abs(0 - 0) + abs(8 - 0) == 8
    assert got[0, 8] == 24
    assert got[0, 8] == ref_bfs(passable, (0, 0))[(0, 8)]


def test_unreachable_cells_are_not_selectable():
    """A sealed pocket keeps the sentinel, so it can never be drawn as a spawn."""
    passable = np.zeros((7, 7), dtype=bool)
    passable[0:3, 0:3] = True     # region with the start
    passable[5:7, 5:7] = True     # sealed-off pocket
    got = np.asarray(bfs_distance_field(jnp.array(passable), (0, 0)))
    assert got[6, 6] == 49        # h*w sentinel
    assert got[2, 2] == 4


# --------------------------------------------------- real competition boards


def test_competition_boards_respect_the_walking_floor():
    """The property that matters, on the grid the evaluator actually plays."""
    env = GeneralsEnv(mode="competition")
    below = []
    for seed in range(200):
        grid = competition_board(env, seed, COMPETITION_MIN_DISTANCE)
        d = walk_distance(grid)
        assert d is not None, f"seed {seed}: generals disconnected"
        if d < COMPETITION_MIN_DISTANCE:
            below.append((seed, d))
    assert not below, f"boards below the floor: {below[:5]}"


def test_stripping_the_castles_cannot_shorten_the_walk():
    """Every terrain decision precedes both spawns, so the distance the
    generator enforced is the distance the match is played at.

    This is the regression that motivated the ordering: when castles were carved
    out of mountains AFTER the generals were placed, a run of them could tunnel
    through the ridge and boards generated at 17+ were played at 8.
    """
    env = GeneralsEnv(mode="competition")
    for seed in range(200):
        grid = competition_board(env, seed, COMPETITION_MIN_DISTANCE)
        stripped = bc.strip_neutral_castles(grid)
        assert walk_distance(stripped) == walk_distance(grid), f"seed {seed}"
        assert walk_distance(stripped) >= COMPETITION_MIN_DISTANCE, f"seed {seed}"


def test_short_straight_lines_are_allowed_when_the_walk_is_long():
    """The point of BFS: boards the Manhattan proxy would have thrown away.

    A straight-line floor of 17 can only ever produce spawns 17+ apart in
    Manhattan terms. BFS admits closer-looking spawns whose route is long.
    """
    env = GeneralsEnv(mode="competition")
    detours = 0
    for seed in range(200):
        grid = competition_board(env, seed, COMPETITION_MIN_DISTANCE)
        a, b = generals_of(grid)
        manhattan = abs(a[0] - b[0]) + abs(a[1] - b[1])
        if manhattan < COMPETITION_MIN_DISTANCE:
            detours += 1
    assert detours > 0, "no board used a detour — BFS is behaving like Manhattan"


def test_generals_are_always_placed_and_connected():
    env = GeneralsEnv(mode="competition")
    for seed in range(100):
        grid = competition_board(env, seed, COMPETITION_MIN_DISTANCE)
        g = np.asarray(grid)
        assert int((g == 1).sum()) == 1 and int((g == 2).sum()) == 1, f"seed {seed}"
        assert walk_distance(grid) is not None


def test_tight_board_falls_back_instead_of_failing():
    """An impossible floor must still yield a legal board (the farthest pair),
    never a crash, a hang, or a missing general."""
    grid = generate_grid(
        jax.random.PRNGKey(3), grid_dims=(8, 8),
        mountain_density_range=(0.24, 0.26), num_castles_range=(2, 3),
        min_generals_distance=500,  # unsatisfiable on an 8x8
        castle_val_range=(20, 26),
    )[:8, :8]
    g = np.asarray(grid)
    assert int((g == 1).sum()) == 1 and int((g == 2).sum()) == 1
    assert walk_distance(grid) is not None
