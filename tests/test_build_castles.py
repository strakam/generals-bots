"""Tests for the build-castles modifier (generals.modifiers.build_castles)."""
import jax.numpy as jnp
import jax.random as jrandom

from generals import GeneralsEnv
from generals.core import game
from generals.modifiers import build_castles as bc

PASS = jnp.array([1, 0, 0, 0, 0], dtype=jnp.int32)


def make_grid(size=8):
    """Empty grid, P0 general at (0,0), P1 general at (size-1, size-1)."""
    grid = jnp.zeros((size, size), dtype=jnp.int32)
    grid = grid.at[0, 0].set(1)
    grid = grid.at[size - 1, size - 1].set(2)
    return grid


def give_cell(state, player, ij, army):
    """Hand a cell (with army) to a player, as if conquered earlier."""
    i, j = ij
    return state._replace(
        armies=state.armies.at[i, j].set(army),
        ownership=state.ownership.at[player, i, j].set(True),
        ownership_neutral=state.ownership_neutral.at[i, j].set(False),
    )


def build_action(i, j):
    return jnp.array([bc.BUILD, i, j, 0, 0], dtype=jnp.int32)


def actions_of(a0, a1=PASS):
    return jnp.stack([a0, a1])


# ---------------------------------------------------------------- distances


def test_distance_field_open_grid_equals_manhattan():
    state = game.create_initial_state(make_grid(8))
    dists = bc.compute_build_distances(state)
    ii, jj = jnp.meshgrid(jnp.arange(8), jnp.arange(8), indexing="ij")
    assert jnp.array_equal(dists[0], ii + jj)  # P0 general at (0,0)
    assert jnp.array_equal(dists[1], (7 - ii) + (7 - jj))  # P1 at (7,7)


def test_distance_field_detours_around_mountains():
    # Wall on column 1, open only at the bottom row.
    grid = make_grid(5)
    for i in range(4):
        grid = grid.at[i, 1].set(-2)
    state = game.create_initial_state(grid)
    dists = bc.compute_build_distances(state)
    # (0,0) -> (0,2): down to row 4, across, back up = 4 + 2 + 4 = 10.
    assert int(dists[0, 0, 2]) == 10
    assert int(dists[0, 4, 1]) == 5  # the gap itself


def test_distance_field_unreachable_pocket():
    # Cell (0,2) sealed off by mountains at (0,1), (1,2) and (0,3).
    grid = make_grid(5)
    grid = grid.at[0, 1].set(-2)
    grid = grid.at[1, 2].set(-2)
    grid = grid.at[0, 3].set(-2)
    state = game.create_initial_state(grid)
    dists = bc.compute_build_distances(state)
    assert int(dists[0, 0, 2]) >= int(bc.UNREACHABLE)


def test_cost_formula():
    dists = jnp.arange(60).reshape(1, 6, 10)
    costs = bc.build_costs(dists)
    assert int(costs[0, 0, 1]) == 55  # d=1, next to the general
    assert int(costs[0, 0, 9]) == 47  # d=9
    assert int(costs[0, 2, 6]) == 30  # d=26, exactly at the clamp
    assert int(costs[0, 5, 9]) == 30  # far away stays clamped


# ------------------------------------------------------------------- builds


def test_build_pays_cost_and_keeps_remainder():
    state = give_cell(game.create_initial_state(make_grid(8)), 0, (0, 1), 60)
    dists = bc.compute_build_distances(state)
    new_state, info = bc.step(state, actions_of(build_action(0, 1)), dists)

    assert bool(new_state.cities[0, 1])
    assert int(new_state.armies[0, 1]) == 60 - 55  # d=1 -> cost 55
    assert bool(new_state.ownership[0, 0, 1])
    assert int(new_state.winner) == -1


def test_build_with_exact_cost_leaves_zero_army_snipeable():
    state = give_cell(game.create_initial_state(make_grid(8)), 0, (0, 1), 55)
    state = give_cell(state, 1, (0, 2), 2)
    dists = bc.compute_build_distances(state)

    state, _ = bc.step(state, actions_of(build_action(0, 1)), dists)
    assert bool(state.cities[0, 1])
    assert int(state.armies[0, 1]) == 0

    # One enemy unit takes the fresh, empty castle.
    snipe = jnp.array([0, 0, 2, 2, 0], dtype=jnp.int32)  # move LEFT from (0,2)
    state, _ = bc.step(state, actions_of(PASS, snipe), dists)
    assert bool(state.ownership[1, 0, 1])
    assert not bool(state.ownership[0, 0, 1])
    assert bool(state.cities[0, 1])  # still a castle, now enemy-owned


def test_build_rejected_when_army_insufficient():
    state = give_cell(game.create_initial_state(make_grid(8)), 0, (0, 1), 54)
    dists = bc.compute_build_distances(state)
    new_state, _ = bc.step(state, actions_of(build_action(0, 1)), dists)
    assert not bool(new_state.cities[0, 1])
    assert int(new_state.armies[0, 1]) == 54


def test_build_cheaper_far_from_general():
    # (4,4) is d=8 from (0,0) -> cost 48.
    state = give_cell(game.create_initial_state(make_grid(8)), 0, (4, 4), 48)
    dists = bc.compute_build_distances(state)
    new_state, _ = bc.step(state, actions_of(build_action(4, 4)), dists)
    assert bool(new_state.cities[4, 4])
    assert int(new_state.armies[4, 4]) == 0


def test_build_rejected_on_unowned_and_enemy_cells():
    state = game.create_initial_state(make_grid(8))
    state = give_cell(state, 1, (3, 3), 100)
    state = state._replace(armies=state.armies.at[2, 2].set(100))  # neutral pile
    dists = bc.compute_build_distances(state)

    for target in [(2, 2), (3, 3)]:  # neutral, enemy-owned
        new_state, _ = bc.step(state, actions_of(build_action(*target)), dists)
        assert not bool(new_state.cities[target])
        assert jnp.array_equal(new_state.armies, state.armies)


def test_build_rejected_on_general_and_existing_castle():
    state = game.create_initial_state(make_grid(8))
    state = state._replace(armies=state.armies.at[0, 0].set(100))
    state = give_cell(state, 0, (0, 1), 200)
    dists = bc.compute_build_distances(state)

    new_state, _ = bc.step(state, actions_of(build_action(0, 0)), dists)
    assert not bool(new_state.cities[0, 0])  # generals can't become castles

    state, _ = bc.step(state, actions_of(build_action(0, 1)), dists)
    assert bool(state.cities[0, 1])
    armies_after_first = int(state.armies[0, 1])
    state, _ = bc.step(state, actions_of(build_action(0, 1)), dists)
    # No double-charge: the only change is the castle's +1 production
    # (time went 1 -> 2, an even tick), not another -55.
    assert int(state.armies[0, 1]) == armies_after_first + 1


def test_build_out_of_bounds_is_a_noop():
    state = give_cell(game.create_initial_state(make_grid(8)), 0, (0, 1), 100)
    dists = bc.compute_build_distances(state)
    for i, j in [(-1, 0), (0, -3), (99, 0), (0, 99)]:
        new_state, _ = bc.step(state, actions_of(build_action(i, j)), dists)
        assert int(new_state.cities.sum()) == 0
        assert jnp.array_equal(new_state.armies[0, 1], state.armies[0, 1])


def test_build_ignored_after_game_over():
    state = give_cell(game.create_initial_state(make_grid(8)), 0, (0, 1), 100)
    state = state._replace(winner=jnp.int32(1))
    dists = bc.compute_build_distances(state)
    new_state, _ = bc.step(state, actions_of(build_action(0, 1)), dists)
    assert not bool(new_state.cities[0, 1])


def test_both_players_build_same_turn():
    state = game.create_initial_state(make_grid(8))
    state = give_cell(state, 0, (0, 1), 60)
    state = give_cell(state, 1, (7, 6), 60)
    dists = bc.compute_build_distances(state)
    new_state, _ = bc.step(state, actions_of(build_action(0, 1), build_action(7, 6)), dists)
    assert bool(new_state.cities[0, 1])
    assert bool(new_state.cities[7, 6])
    assert int(new_state.armies[0, 1]) == 5
    assert int(new_state.armies[7, 6]) == 5


# ------------------------------------------- interaction with the base game


def test_non_build_actions_match_base_step_exactly():
    state = give_cell(game.create_initial_state(make_grid(8)), 0, (0, 1), 10)
    dists = bc.compute_build_distances(state)
    move = jnp.array([0, 0, 1, 3, 0], dtype=jnp.int32)  # (0,1) RIGHT, all-in

    base_state, base_info = game.step(state, actions_of(move))
    mod_state, mod_info = bc.step(state, actions_of(move), dists)

    for base_field, mod_field in zip(base_state, mod_state):
        assert jnp.array_equal(base_field, mod_field)
    assert jnp.array_equal(base_info.army, mod_info.army)


def test_built_castle_produces_like_a_city():
    state = give_cell(game.create_initial_state(make_grid(8)), 0, (0, 1), 60)
    dists = bc.compute_build_distances(state)
    state, _ = bc.step(state, actions_of(build_action(0, 1)), dists)  # time 0 -> 1
    assert int(state.armies[0, 1]) == 5

    passes = actions_of(PASS)
    state, _ = bc.step(state, passes, dists)  # time -> 2 (even: +1)
    assert int(state.armies[0, 1]) == 6
    state, _ = bc.step(state, passes, dists)  # time -> 3 (odd: nothing)
    assert int(state.armies[0, 1]) == 6
    state, _ = bc.step(state, passes, dists)  # time -> 4 (even: +1)
    assert int(state.armies[0, 1]) == 7


# --------------------------------------------------------------- env wiring


def test_env_flag_strips_neutral_castles():
    key = jrandom.PRNGKey(7)
    params = dict(grid_dims=(12, 12), truncation=100, num_cities_range=(9, 11))
    with_castles = GeneralsEnv(**params).init_state(key)
    without = GeneralsEnv(**params, build_castles=True).init_state(key)

    assert int(with_castles.cities.sum()) >= 2  # generator always places 2+
    assert int(without.cities.sum()) == 0
    # Stripped city cells become plain neutral ground, not mountains.
    assert int(without.mountains.sum()) <= int(with_castles.mountains.sum())


def test_env_step_applies_build():
    env = GeneralsEnv(grid_dims=(8, 8), truncation=100, num_cities_range=(9, 11),
                      mountain_density_range=(0.0, 0.0), build_castles=True)
    key = jrandom.PRNGKey(3)
    pool, state = env.reset(key)

    gi, gj = (int(x) for x in state.general_positions[0])
    # Give P0 a buildable cell next to nothing in particular.
    ti, tj = (gi + 1) % 8, gj
    state = give_cell(state, 0, (ti, tj), 80)

    timestep, new_state = env.step(state, actions_of(build_action(ti, tj)), pool)
    assert bool(new_state.cities[ti, tj])
    assert bool(timestep.observation.cities[0][ti, tj])  # visible to the builder


def test_full_game_with_builder_policy_runs():
    """A scripted builder plays 200 turns; the game must stay well-formed."""
    state = game.create_initial_state(make_grid(8))
    dists = bc.compute_build_distances(state)
    build_site = (0, 1)
    built = False

    for _ in range(200):
        if not built and int(state.armies[0, 0]) > 56:
            a0 = jnp.array([0, 0, 0, 3, 0], dtype=jnp.int32)  # move the stack RIGHT onto (0,1)
        elif not built and int(state.armies[build_site]) >= 55 and bool(state.ownership[0][build_site]):
            a0 = build_action(*build_site)
        else:
            a0 = PASS
        state, info = bc.step(state, actions_of(a0), dists)
        if bool(state.cities[build_site]):
            built = True
        if bool(info.is_done):
            break

    assert built, "builder never managed to build a castle"
    assert int(state.cities.sum()) == 1
    assert int(state.winner) == -1  # nobody died from passing around
