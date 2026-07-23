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


def give_castle(state, player, ij):
    """Plant a finished castle owned by `player` at ij."""
    state = give_cell(state, player, ij, 1)
    return state._replace(castles=state.castles.at[ij].set(True))


def build_action(i, j):
    return jnp.array([bc.BUILD, i, j, 0, 0], dtype=jnp.int32)


def actions_of(a0, a1=PASS):
    return jnp.stack([a0, a1])


# ------------------------------------------------------------------ pricing


def test_cost_grid_base_and_general_proximity():
    # Only structure: P0 general at (0,0). cost = 35 + max(0, 10 - 2d).
    state = game.create_initial_state(make_grid(8))
    costs = bc.build_cost_grid(state, 0)
    assert int(costs[0, 1]) == 43  # d=1
    assert int(costs[1, 1]) == 41  # d=2
    assert int(costs[0, 3]) == 39  # d=3
    assert int(costs[2, 2]) == 37  # d=4
    assert int(costs[0, 5]) == 35  # d=5: surcharge gone
    assert int(costs[6, 6]) == 35  # far away


def test_cost_grid_sums_over_own_structures():
    state = game.create_initial_state(make_grid(12))
    state = give_castle(state, 0, (0, 4))
    state = give_castle(state, 0, (4, 0))
    costs = bc.build_cost_grid(state, 0)
    # (0,2): general d=2 (+6), castle(0,4) d=2 (+6), castle(4,0) d=6 (+0).
    assert int(costs[0, 2]) == 35 + 6 + 6
    # (2,2): general d=4 (+2), both castles d=4 (+2 each).
    assert int(costs[2, 2]) == 35 + 2 + 2 + 2
    # (9,9): everything is far.
    assert int(costs[9, 9]) == 35


def test_cost_grid_ignores_enemy_structures():
    state = game.create_initial_state(make_grid(8))
    baseline = bc.build_cost_grid(state, 0)
    state = give_castle(state, 1, (2, 2))  # enemy castle
    costs = bc.build_cost_grid(state, 0)
    assert jnp.array_equal(costs, baseline)
    # Enemy general contributes nothing either: cells near (7,7) stay at base.
    assert int(costs[7, 6]) == 35


def test_captured_castle_raises_your_prices():
    state = game.create_initial_state(make_grid(8))
    state = give_castle(state, 1, (4, 4))
    before = int(bc.build_cost_grid(state, 0)[4, 5])
    # P0 takes the castle: ownership flips, structure now counts as P0's.
    state = state._replace(
        ownership=state.ownership.at[1, 4, 4].set(False).at[0, 4, 4].set(True))
    after = int(bc.build_cost_grid(state, 0)[4, 5])
    assert before == 35
    assert after == 43  # d=1 from the newly owned castle


# ------------------------------------------------------------------- builds


def test_build_pays_cost_and_keeps_remainder():
    state = give_cell(game.create_initial_state(make_grid(8)), 0, (0, 1), 60)
    new_state, info = bc.step(state, actions_of(build_action(0, 1)))

    assert bool(new_state.castles[0, 1])
    assert int(new_state.armies[0, 1]) == 60 - 43  # next to the general
    assert bool(new_state.ownership[0, 0, 1])
    assert int(new_state.winner) == -1


def test_build_with_exact_cost_leaves_zero_army_snipeable():
    state = give_cell(game.create_initial_state(make_grid(8)), 0, (0, 1), 43)
    state = give_cell(state, 1, (0, 2), 2)

    state, _ = bc.step(state, actions_of(build_action(0, 1)))
    assert bool(state.castles[0, 1])
    assert int(state.armies[0, 1]) == 0

    # One enemy unit takes the fresh, empty castle.
    snipe = jnp.array([0, 0, 2, 2, 0], dtype=jnp.int32)  # move LEFT from (0,2)
    state, _ = bc.step(state, actions_of(PASS, snipe))
    assert bool(state.ownership[1, 0, 1])
    assert not bool(state.ownership[0, 0, 1])
    assert bool(state.castles[0, 1])  # still a castle, now enemy-owned


def test_build_rejected_when_army_insufficient():
    state = give_cell(game.create_initial_state(make_grid(8)), 0, (0, 1), 42)
    new_state, _ = bc.step(state, actions_of(build_action(0, 1)))
    assert not bool(new_state.castles[0, 1])
    assert int(new_state.armies[0, 1]) == 42


def test_build_costs_base_price_away_from_structures():
    # (4,4) is d=8 from the general: no surcharge, exactly 35.
    state = give_cell(game.create_initial_state(make_grid(8)), 0, (4, 4), 35)
    new_state, _ = bc.step(state, actions_of(build_action(4, 4)))
    assert bool(new_state.castles[4, 4])
    assert int(new_state.armies[4, 4]) == 0


def test_prices_rise_as_you_build():
    # First castle at (0,1) makes (0,3) pricier: 35 + general d=3 (+4) + castle d=2 (+6).
    state = game.create_initial_state(make_grid(8))
    state = give_cell(state, 0, (0, 1), 43)
    state = give_cell(state, 0, (0, 3), 44)
    state, _ = bc.step(state, actions_of(build_action(0, 1)))
    assert bool(state.castles[0, 1])

    short_state, _ = bc.step(state, actions_of(build_action(0, 3)))
    assert not bool(short_state.castles[0, 3])  # 44 < 45 now

    state = state._replace(armies=state.armies.at[0, 3].set(45))
    state, _ = bc.step(state, actions_of(build_action(0, 3)))
    assert bool(state.castles[0, 3])
    # Paid 45 exactly, then time hit 2 (even tick) and the new castle produced.
    assert int(state.armies[0, 3]) == 1


def test_build_rejected_on_unowned_and_enemy_cells():
    state = game.create_initial_state(make_grid(8))
    state = give_cell(state, 1, (3, 3), 100)
    state = state._replace(armies=state.armies.at[2, 2].set(100))  # neutral pile

    for target in [(2, 2), (3, 3)]:  # neutral, enemy-owned
        new_state, _ = bc.step(state, actions_of(build_action(*target)))
        assert not bool(new_state.castles[target])
        assert jnp.array_equal(new_state.armies, state.armies)


def test_build_rejected_on_general_and_existing_castle():
    state = game.create_initial_state(make_grid(8))
    state = state._replace(armies=state.armies.at[0, 0].set(100))
    state = give_cell(state, 0, (0, 1), 200)

    new_state, _ = bc.step(state, actions_of(build_action(0, 0)))
    assert not bool(new_state.castles[0, 0])  # generals can't become castles

    state, _ = bc.step(state, actions_of(build_action(0, 1)))
    assert bool(state.castles[0, 1])
    armies_after_first = int(state.armies[0, 1])
    state, _ = bc.step(state, actions_of(build_action(0, 1)))
    # No double-charge: the only change is the castle's +1 production
    # (time went 1 -> 2, an even tick), not another price deduction.
    assert int(state.armies[0, 1]) == armies_after_first + 1


def test_build_out_of_bounds_is_a_noop():
    state = give_cell(game.create_initial_state(make_grid(8)), 0, (0, 1), 100)
    for i, j in [(-1, 0), (0, -3), (99, 0), (0, 99)]:
        new_state, _ = bc.step(state, actions_of(build_action(i, j)))
        assert int(new_state.castles.sum()) == 0
        assert jnp.array_equal(new_state.armies[0, 1], state.armies[0, 1])


def test_build_ignored_after_game_over():
    state = give_cell(game.create_initial_state(make_grid(8)), 0, (0, 1), 100)
    state = state._replace(winner=jnp.int32(1))
    new_state, _ = bc.step(state, actions_of(build_action(0, 1)))
    assert not bool(new_state.castles[0, 1])


def test_both_players_build_same_turn():
    state = game.create_initial_state(make_grid(8))
    state = give_cell(state, 0, (0, 1), 60)
    state = give_cell(state, 1, (7, 6), 60)
    new_state, _ = bc.step(state, actions_of(build_action(0, 1), build_action(7, 6)))
    assert bool(new_state.castles[0, 1])
    assert bool(new_state.castles[7, 6])
    assert int(new_state.armies[0, 1]) == 60 - 43
    assert int(new_state.armies[7, 6]) == 60 - 43


# ------------------------------------------- interaction with the base game


def test_non_build_actions_match_base_step_exactly():
    state = give_cell(game.create_initial_state(make_grid(8)), 0, (0, 1), 10)
    move = jnp.array([0, 0, 1, 3, 0], dtype=jnp.int32)  # (0,1) RIGHT, all-in

    base_state, base_info = game.step(state, actions_of(move))
    mod_state, mod_info = bc.step(state, actions_of(move))

    for base_field, mod_field in zip(base_state, mod_state):
        assert jnp.array_equal(base_field, mod_field)
    assert jnp.array_equal(base_info.army, mod_info.army)


def test_built_castle_produces_like_a_castle():
    state = give_cell(game.create_initial_state(make_grid(8)), 0, (0, 1), 60)
    state, _ = bc.step(state, actions_of(build_action(0, 1)))  # time 0 -> 1
    assert int(state.armies[0, 1]) == 17

    passes = actions_of(PASS)
    state, _ = bc.step(state, passes)  # time -> 2 (even: +1)
    assert int(state.armies[0, 1]) == 18
    state, _ = bc.step(state, passes)  # time -> 3 (odd: nothing)
    assert int(state.armies[0, 1]) == 18
    state, _ = bc.step(state, passes)  # time -> 4 (even: +1)
    assert int(state.armies[0, 1]) == 19


# --------------------------------------------------------------- env wiring


def test_env_flag_strips_neutral_castles():
    key = jrandom.PRNGKey(7)
    params = dict(grid_dims=(12, 12), truncation=100, num_castles_range=(9, 11))
    with_castles = GeneralsEnv(**params).init_state(key)
    without = GeneralsEnv(**params, build_castles=True).init_state(key)

    assert int(with_castles.castles.sum()) >= 2  # generator always places 2+
    assert int(without.castles.sum()) == 0
    # Stripped castle cells become plain neutral ground, not mountains.
    assert int(without.mountains.sum()) <= int(with_castles.mountains.sum())


def test_env_step_applies_build():
    env = GeneralsEnv(grid_dims=(8, 8), truncation=100, num_castles_range=(9, 11),
                      mountain_density_range=(0.0, 0.0), build_castles=True)
    key = jrandom.PRNGKey(3)
    pool, state = env.reset(key)

    gi, gj = (int(x) for x in state.general_positions[0])
    # Give P0 a buildable cell next to its general (cost 43 there).
    ti, tj = (gi + 1) % 8, gj
    state = give_cell(state, 0, (ti, tj), 80)

    timestep, new_state = env.step(state, actions_of(build_action(ti, tj)), pool)
    assert bool(new_state.castles[ti, tj])
    assert bool(timestep.observation.castles[0][ti, tj])  # visible to the builder


def test_full_game_with_builder_policy_runs():
    """A scripted builder plays 200 turns; the game must stay well-formed."""
    state = game.create_initial_state(make_grid(8))
    build_site = (0, 1)
    built = False

    for _ in range(200):
        if not built and int(state.armies[0, 0]) > 50:
            a0 = jnp.array([0, 0, 0, 3, 0], dtype=jnp.int32)  # move the stack RIGHT onto (0,1)
        elif (not built and bool(state.ownership[0][build_site])
              and int(state.armies[build_site]) >= 43):
            a0 = build_action(*build_site)
        else:
            a0 = PASS
        state, info = bc.step(state, actions_of(a0))
        if bool(state.castles[build_site]):
            built = True
        if bool(info.is_done):
            break

    assert built, "builder never managed to build a castle"
    assert int(state.castles.sum()) == 1
    assert int(state.winner) == -1  # nobody died from passing around
