"""Simultaneous decapitation is a draw at every turn.

The base game resolves the two moves in sequence and lets the second capture
overwrite `winner`, so bare game.step awards a mutual decapitation to whoever
moved second. Move order is choosable (game._determine_move_order), so that
made "arrange to move second" a way to win a race the rules call a draw. The
deathtouch modifier — the outermost transition in every competition match —
reports the draw instead, on turn 5 exactly as on turn 900.
"""
import jax.numpy as jnp
import pytest

from generals.core import game
from generals.core.env import GeneralsEnv
from generals.modifiers import deathtouch as dt

PASS = jnp.array([1, 0, 0, 0, 0], dtype=jnp.int32)
COMPETITION_DEATHTOUCH_TURN = 800
UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3


def board(size=8, time=0):
    """P0 general at (0,0), P1 general at (0,size-1), open ground between."""
    grid = jnp.zeros((size, size), dtype=jnp.int32).at[0, 0].set(1).at[0, size - 1].set(2)
    return game.create_initial_state(grid)._replace(time=jnp.int32(time))


def give(state, player, ij, army):
    i, j = ij
    return state._replace(
        armies=state.armies.at[i, j].set(army),
        ownership=state.ownership.at[player, i, j].set(True),
        ownership_neutral=state.ownership_neutral.at[i, j].set(False),
    )


def mutual(time, garrison=1, attacker=40):
    """Both players poised one step from the enemy general, each strong enough
    to capture it outright. P0 strikes (0,7) from (0,6); P1 strikes (0,0) from
    (1,0) — different source tiles, so neither move invalidates the other."""
    s = board(time=time)
    s = s._replace(armies=s.armies.at[0, 0].set(garrison).at[0, 7].set(garrison))
    s = give(s, 0, (0, 6), attacker)
    s = give(s, 1, (1, 0), attacker)
    a0 = jnp.array([0, 0, 6, RIGHT, 0], dtype=jnp.int32)  # (0,6) -> (0,7)
    a1 = jnp.array([0, 1, 0, UP, 0], dtype=jnp.int32)     # (1,0) -> (0,0)
    return s, jnp.stack([a0, a1])


@pytest.mark.parametrize("time", [0, 5, 300, 799, 800, 900])
def test_mutual_capture_is_a_draw_at_every_turn(time):
    s, actions = mutual(time)
    _, info = dt.step(s, actions, COMPETITION_DEATHTOUCH_TURN)
    assert bool(info.is_done), "a mutual decapitation must end the game"
    assert int(info.winner) == -1, f"turn {time}: expected a draw, got player {int(info.winner)}"


def test_base_game_alone_still_picks_the_second_mover():
    """Pins the behaviour we are wrapping (and why the fix cannot live in the
    modifier's threshold branch): bare game.step names a winner pre-threshold."""
    s, actions = mutual(300)
    _, info = game.step(s, actions)
    assert int(info.winner) >= 0


@pytest.mark.parametrize("time", [300, 900])
def test_single_capture_is_unaffected(time):
    """Only P0 strikes; P1 sits still. Still a normal win, not a draw."""
    s, actions = mutual(time)
    actions = actions.at[1].set(PASS)
    _, info = dt.step(s, actions, COMPETITION_DEATHTOUCH_TURN)
    assert bool(info.is_done) and int(info.winner) == 0


def test_quiet_turn_is_untouched():
    """No captures at all: the modifier stays a passthrough."""
    s = give(board(time=300), 0, (4, 4), 10)
    actions = jnp.stack([jnp.array([0, 4, 4, RIGHT, 0], dtype=jnp.int32), PASS])
    got_state, got_info = dt.step(s, actions, COMPETITION_DEATHTOUCH_TURN)
    want_state, want_info = game.step(s, actions)
    for a, b in zip(got_state, want_state):
        assert jnp.array_equal(a, b)
    assert bool(got_info.is_done) == bool(want_info.is_done)
    assert int(got_info.winner) == int(want_info.winner)


def test_mutual_touch_below_threshold_without_the_army_is_not_a_draw():
    """A touch needs no army only from turn 800. Before that, two moves onto
    generals they cannot capture are ordinary failed attacks — play continues."""
    s, actions = mutual(300, garrison=50, attacker=10)  # 9 army vs a 50 garrison
    _, info = dt.step(s, actions, COMPETITION_DEATHTOUCH_TURN)
    assert not bool(info.is_done)
    assert int(info.winner) == -1


def test_competition_preset_wires_the_threshold():
    env = GeneralsEnv(mode="competition")
    assert env.deathtouch_turn == COMPETITION_DEATHTOUCH_TURN


def _moved(army, split):
    """game._execute_move's army arithmetic: half, or all-but-one."""
    m = army // 2 if split else army - 1
    return max(0, min(m, army - 1))


def test_draw_fires_exactly_when_both_generals_fall():
    """Sweep the edge of the rule against an independent oracle.

    Both players stand beside the enemy general with varying garrisons, stack
    sizes, split flags and turns, so the sweep covers real mutual captures,
    near-misses where only one side is strong enough, exact ties (which favour
    the defender), and 1-army stacks with split=0 that move nothing at all.

    The oracle is plain Python capture arithmetic, deliberately NOT a reading of
    the resulting state: game.step transfers the loser's cells to the winner
    before this modifier relabels the outcome, so on a mutual capture the spoils
    hand a general tile back and board ownership no longer shows what happened.
    """
    checked = draws = 0
    for n in (6, 9):
        for garrison in (1, 5, 20):
            for stack in (1, 2, 21, 60):
                for split in (0, 1):
                    for time in (0, 300, 799, 900):
                        s = board(n, time)
                        s = s._replace(armies=s.armies.at[0, 0].set(garrison)
                                                        .at[0, n - 1].set(garrison))
                        s = give(s, 0, (0, n - 2), stack)   # strikes right onto (0,n-1)
                        s = give(s, 1, (1, 0), stack)       # strikes up onto (0,0)
                        actions = jnp.stack([
                            jnp.array([0, 0, n - 2, RIGHT, split], dtype=jnp.int32),
                            jnp.array([0, 1, 0, UP, split], dtype=jnp.int32)])

                        m = _moved(stack, split)
                        valid = m > 0
                        captures = valid and m > garrison      # strict: ties hold
                        touches = valid and time >= COMPETITION_DEATHTOUCH_TURN
                        expect_draw = captures or touches      # symmetric setup

                        _, info = dt.step(s, actions, COMPETITION_DEATHTOUCH_TURN)
                        checked += 1
                        if expect_draw:
                            draws += 1
                            assert int(info.winner) == -1 and bool(info.is_done), (
                                f"n={n} garrison={garrison} stack={stack} split={split} "
                                f"time={time}: expected a draw, got {int(info.winner)}")
                        else:
                            assert int(info.winner) == -1 and not bool(info.is_done), (
                                f"n={n} garrison={garrison} stack={stack} split={split} "
                                f"time={time}: expected play to continue")
    assert checked == 192 and draws > 0
