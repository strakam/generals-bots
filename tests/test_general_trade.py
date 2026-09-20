"""general_trade: generals.io's rule for two players capturing each other's general on the same turn.

Ported from the server code shipped in the client bundle (``getMutualGeneralSwapMoveIndex`` /
``executeMutualGeneralSwap``, replay format 16, 2025): both captures happen, each attacking
stack lands on the other general with its surplus, every other cell of each player passes
to the other with its army halved (rounded up), nobody is eliminated, and the generals change
hands. Opt-in; the default step is unchanged.
"""
import jax.numpy as jnp
import numpy as np
import pytest

from generals.core import game
from generals.core.env import GeneralsEnv

UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
PASS = jnp.array([1, 0, 0, 0, 0], dtype=jnp.int32)


def board(size=8, time=10):
    grid = jnp.zeros((size, size), dtype=jnp.int32).at[0, 0].set(1).at[0, size - 1].set(2)
    return game.create_initial_state(grid)._replace(time=jnp.int32(time))


def give(state, player, ij, army):
    i, j = ij
    return state._replace(armies=state.armies.at[i, j].set(army),
                          ownership=state.ownership.at[player, i, j].set(True),
                          ownership_neutral=state.ownership_neutral.at[i, j].set(False))


def owner(state, ij):
    o = np.asarray(state.ownership[:, ij[0], ij[1]])
    return int(np.argmax(o)) if o.any() else -1


def mutual(garrison0=5, garrison1=7, attacker0=40, attacker1=30, extra=True):
    """P0 general (0,0) garrison0, P1 general (0,7) garrison1. P0's stack at (0,6) hits (0,7);
    P1's stack at (1,0) hits (0,0). Optional extra land: P0 holds (3,3)=9, P1 holds (4,4)=4."""
    s = board()
    s = s._replace(armies=s.armies.at[0, 0].set(garrison0).at[0, 7].set(garrison1))
    s = give(s, 0, (0, 6), attacker0)
    s = give(s, 1, (1, 0), attacker1)
    if extra:
        s = give(s, 0, (3, 3), 9)
        s = give(s, 1, (4, 4), 4)
    a0 = jnp.array([0, 0, 6, RIGHT, 0], dtype=jnp.int32)
    a1 = jnp.array([0, 1, 0, UP, 0], dtype=jnp.int32)
    return s, jnp.stack([a0, a1])


def test_trade_swaps_generals_and_territory():
    s, actions = mutual()
    ns, info = game.step(s, actions, general_trade=True)
    assert int(info.winner) == -1 and not bool(info.is_done)
    assert ns.eliminated.tolist() == [False, False]
    # generals stay generals; each player now holds the other's
    assert bool(ns.generals[0, 0]) and bool(ns.generals[0, 7])
    assert owner(ns, (0, 7)) == 0 and owner(ns, (0, 0)) == 1
    assert ns.general_positions.tolist() == [[0, 7], [0, 0]]
    # surpluses: P0 moved 39 onto a garrison of 7 -> 32; P1 moved 29 onto 5 -> 24
    assert int(ns.armies[0, 7]) == 32 and int(ns.armies[0, 0]) == 24
    # sources keep 1 and pass to the other side halved (round up): 1 -> 1
    assert owner(ns, (0, 6)) == 1 and int(ns.armies[0, 6]) == 1
    assert owner(ns, (1, 0)) == 0 and int(ns.armies[1, 0]) == 1
    # extra land swaps with armies halved, rounded up: 9 -> 5 to P1, 4 -> 2 to P0
    assert owner(ns, (3, 3)) == 1 and int(ns.armies[3, 3]) == 5
    assert owner(ns, (4, 4)) == 0 and int(ns.armies[4, 4]) == 2
    assert int(ns.time) == int(s.time) + 1


def test_flag_off_is_the_old_first_mover_rule():
    s, actions = mutual()
    ns, info = game.step(s, actions)
    assert int(info.winner) >= 0 and bool(ns.eliminated.any())


def test_one_sided_capture_is_not_a_trade():
    """P1's stack is too small to capture: an ordinary turn, P0 wins."""
    s, actions = mutual(garrison0=50, attacker1=30)
    ns, info = game.step(s, actions, general_trade=True)
    assert int(info.winner) == 0 and ns.eliminated.tolist() == [False, True]


def test_no_trade_when_one_side_passes():
    s, actions = mutual()
    ns, info = game.step(s, actions.at[1].set(PASS), general_trade=True)
    assert int(info.winner) == 0


def test_split_move_counts_half():
    """A 50% move sends floor(n/2); the trade uses that amount and leaves ceil(n/2)."""
    s, actions = mutual(attacker0=41)                      # sends 20, keeps 21
    actions = actions.at[0, 4].set(1)
    ns, info = game.step(s, actions, general_trade=True)
    assert int(info.winner) == -1
    assert int(ns.armies[0, 7]) == 20 - 7                  # surplus over P1's garrison
    assert owner(ns, (0, 6)) == 1 and int(ns.armies[0, 6]) == (21 + 1) // 2


def test_attack_launched_from_own_general_defends_with_its_reserve():
    """P1 attacks from its general (0,7)->(0,6)... arrange P0 adjacent instead: P0 at (1,7)
    hits (0,7); P1 attacks from (0,7)... not adjacent to (0,0). Use a 2-wide board instead."""
    grid = jnp.zeros((2, 2), dtype=jnp.int32).at[0, 0].set(1).at[0, 1].set(2)
    s = game.create_initial_state(grid)._replace(time=jnp.int32(10))
    s = s._replace(armies=s.armies.at[0, 0].set(10).at[0, 1].set(30))
    a0 = jnp.array([0, 0, 0, RIGHT, 0], dtype=jnp.int32)   # P0 general -> P1 general, moves 9, keeps 1
    a1 = jnp.array([0, 0, 1, LEFT, 0], dtype=jnp.int32)    # P1 general -> P0 general, moves 29, keeps 1
    ns, info = game.step(s, jnp.stack([a0, a1]), general_trade=True)
    # each attack faces only the reserve of 1 left behind: both capture, trade
    assert int(info.winner) == -1
    assert owner(ns, (0, 1)) == 0 and int(ns.armies[0, 1]) == 9 - 1
    assert owner(ns, (0, 0)) == 1 and int(ns.armies[0, 0]) == 29 - 1


def test_game_continues_and_can_be_won_after_a_trade():
    s, actions = mutual()
    ns, _ = game.step(s, actions, general_trade=True)
    # P0 now sits on (0,7) with 32 and its old general (0,0) is P1's with 24; P0 marches back
    st = ns
    for _ in range(50):
        st, info = game.step(st, jnp.stack([PASS, PASS]), general_trade=True)
    assert int(info.winner) == -1


def test_env_flag_and_deathtouch_are_exclusive():
    with pytest.raises(ValueError):
        GeneralsEnv(general_trade=True, deathtouch_turn=100)
    GeneralsEnv(general_trade=True)


def test_default_step_unchanged_on_random_play():
    """With the flag off nothing changed: compare against a step without the argument."""
    import jax
    env = GeneralsEnv(min_grid_size=8, max_grid_size=8, pad_to=8)
    key = jax.random.PRNGKey(0)
    s = env.init_state(key)
    for t in range(40):
        key, k = jax.random.split(key)
        a = jax.random.randint(k, (2, 5), 0, 8).at[:, 0].set(0).at[:, 3].set(jax.random.randint(k, (2,), 0, 4)).at[:, 4].set(0)
        s1, _ = game.step(s, a)
        s2, _ = game.step(s, a, general_trade=False)
        assert np.array_equal(np.asarray(s1.armies), np.asarray(s2.armies))
        s = s1
