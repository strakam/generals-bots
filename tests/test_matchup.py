"""The local match runner must play the ruleset it says it plays.

`competition/matchup.py --mode competition` is how competitors rehearse against
the real format, so its per-turn transition has to compose the same modifiers
the evaluator does. `game.step` alone silently plays a *different* game rather
than failing: it tests `pass == 1`, so a build action falls through to a plain
move, and deathtouch never fires. These tests pin the composition.
"""
import sys
from pathlib import Path

import jax.numpy as jnp
import pytest

from generals.core import game
from generals.core.env import GeneralsEnv
from generals.core.game import create_initial_state

# matchup.py imports `protocol` as a top-level module, the way it resolves when
# run as `python competition/matchup.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "competition"))
import matchup  # noqa: E402

PASS = jnp.array([1, 0, 0, 0, 0], dtype=jnp.int32)
COMPETITION_DEATHTOUCH_TURN = 800


def open_board(size=8, time=0):
    """Open board, P0 general at (0,0), P1 general at (0, size-1)."""
    grid = jnp.zeros((size, size), dtype=jnp.int32).at[0, 0].set(1).at[0, size - 1].set(2)
    return create_initial_state(grid)._replace(time=jnp.int32(time))


def give(state, player, ij, army):
    i, j = ij
    return state._replace(
        armies=state.armies.at[i, j].set(army),
        ownership=state.ownership.at[player, i, j].set(True),
        ownership_neutral=state.ownership_neutral.at[i, j].set(False),
    )


def test_competition_mode_builds_castles():
    """A funded build under --mode competition must actually build."""
    transition = matchup.make_transition(GeneralsEnv(mode="competition"))
    # (5,5) is 10 steps from P0's general, past the d>=5 proximity surcharge,
    # so the price is the base 35.
    state = give(open_board(), 0, (5, 5), 60)

    build = jnp.array([2, 5, 5, 0, 0], dtype=jnp.int32)
    state, _ = transition(state, jnp.stack([build, PASS]))

    assert bool(state.castles[5, 5]), "build action was not applied"
    assert int(state.armies[5, 5]) == 60 - 35, "cost was not deducted"
    # Regression: game.step alone reads pass==2 as "not a pass" and moves the
    # army out of the cell instead (leaving 1 behind, 59 one row up).
    assert int(state.armies[4, 5]) == 0, "the build leaked out as a move"


def test_competition_mode_fires_deathtouch():
    """Past turn 800, stepping onto the enemy general must win outright."""
    transition = matchup.make_transition(GeneralsEnv(mode="competition"))
    state = open_board(time=COMPETITION_DEATHTOUCH_TURN + 1)
    state = give(state, 1, (0, 7), 500)         # enemy general, unassailable by force
    state = give(state, 0, (0, 6), 3)           # 3 army next to it

    touch = jnp.array([0, 0, 6, 3, 0], dtype=jnp.int32)   # move right, onto (0,7)
    _, info = transition(state, jnp.stack([touch, PASS]))

    assert bool(info.is_done) and int(info.winner) == 0


def test_deathtouch_does_not_fire_early():
    """The same touch before turn 800 is an ordinary (failed) attack."""
    transition = matchup.make_transition(GeneralsEnv(mode="competition"))
    state = open_board(time=COMPETITION_DEATHTOUCH_TURN - 1)
    state = give(state, 1, (0, 7), 500)
    state = give(state, 0, (0, 6), 3)

    touch = jnp.array([0, 0, 6, 3, 0], dtype=jnp.int32)
    _, info = transition(state, jnp.stack([touch, PASS]))

    assert not bool(info.is_done)


def test_plain_env_transition_is_the_base_game():
    """Without modifiers the transition stays exactly game.step."""
    env = GeneralsEnv(grid_dims=(8, 8), truncation=100)
    assert not env.build_castles and env.deathtouch_turn is None
    transition = matchup.make_transition(env)
    state = give(open_board(), 0, (2, 2), 60)

    move = jnp.array([0, 2, 2, 1, 0], dtype=jnp.int32)
    got, _ = transition(state, jnp.stack([move, PASS]))
    want, _ = game.step(state, jnp.stack([move, PASS]))

    assert jnp.array_equal(got.armies, want.armies)
    assert jnp.array_equal(got.ownership, want.ownership)


@pytest.mark.parametrize("seed", [0, 1, 7])
def test_competition_boards_are_rectangular_and_castle_free(seed):
    """The runner must draw the boards the evaluator plays: exact rectangles in
    [18, 21] (not a 22x22 pad_to square) with no neutral castles to capture."""
    env = GeneralsEnv(mode="competition")
    state = matchup.make_board(env, seed)
    h, w = state.armies.shape

    assert env.min_grid_size <= h <= env.max_grid_size
    assert env.min_grid_size <= w <= env.max_grid_size
    assert int(state.castles.sum()) == 0, "neutral castles were not stripped"
    assert int(state.generals.sum()) == 2
