"""legacy_move_priority: the pre-2025 alternating first-mover rule, opt-in only.

Default behaviour must be byte-for-byte the current rule (chasing > reinforcing
> smaller army); with the flag on, player 0 resolves first on even ticks and
player 1 on odd ticks regardless of what the moves are.
"""
import jax.numpy as jnp
import numpy as np

from generals.core import game
from generals.core.env import GeneralsEnv

UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3


def board(size=8, time=0):
    grid = jnp.zeros((size, size), dtype=jnp.int32).at[0, 0].set(1).at[0, size - 1].set(2)
    return game.create_initial_state(grid)._replace(time=jnp.int32(time))


def give(state, player, ij, army):
    i, j = ij
    return state._replace(
        armies=state.armies.at[i, j].set(army),
        ownership=state.ownership.at[player, i, j].set(True),
        ownership_neutral=state.ownership_neutral.at[i, j].set(False),
    )


def contested(time):
    """P0 (25) and P1 (40) both move onto the neutral garrison (2,2) of 20."""
    s = board(time=time)
    s = give(s, 0, (2, 1), 25)
    s = give(s, 1, (2, 3), 40)
    s = s._replace(armies=s.armies.at[2, 2].set(20))
    a0 = jnp.array([0, 2, 1, RIGHT, 0], dtype=jnp.int32)
    a1 = jnp.array([0, 2, 3, LEFT, 0], dtype=jnp.int32)
    return s, jnp.stack([a0, a1])


def test_default_is_unchanged():
    for time in (0, 1, 2, 3):
        s, a = contested(time)
        assert np.array_equal(game._determine_move_order(s, a), game._determine_move_order(s, a, False))
        s_default, info_default = game.step(s, a)
        s_explicit, info_explicit = game.step(s, a, legacy_move_priority=False)
        for x, y in zip(s_default, s_explicit):
            assert np.array_equal(np.asarray(x), np.asarray(y))
        # smaller army first, so the 40 resolves last and holds the cell on every tick
        assert bool(s_default.ownership[1, 2, 2]) and int(s_default.armies[2, 2]) == 35


def test_legacy_alternates_by_tick():
    for time in range(6):
        s, a = contested(time)
        order = np.asarray(game._determine_move_order(s, a, legacy_move_priority=True))
        expected = [0, 1] if time % 2 == 0 else [1, 0]
        assert order.tolist() == expected, (time, order)


def test_legacy_changes_who_holds_a_contested_cell_on_odd_ticks():
    # even tick: P0 (24 moved) clears the 20-garrison first, P1's 39 takes it: 39-4 = 35
    s, a = contested(0)
    s2, _ = game.step(s, a, legacy_move_priority=True)
    assert bool(s2.ownership[1, 2, 2]) and int(s2.armies[2, 2]) == 35
    # odd tick: P1's 39 clears the garrison first (holds 19), then P0's 24 beats 19: P0 holds 5
    s, a = contested(1)
    s2, _ = game.step(s, a, legacy_move_priority=True)
    assert bool(s2.ownership[0, 2, 2]) and int(s2.armies[2, 2]) == 5


def test_env_flag_is_plumbed_and_off_by_default():
    assert GeneralsEnv(grid_dims=(8, 8), pool_size=2).legacy_move_priority is False
    assert GeneralsEnv(grid_dims=(8, 8), pool_size=2, legacy_move_priority=True).legacy_move_priority is True
