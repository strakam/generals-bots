"""Deathtouch modifier tests — the touch must mirror the engine's own
move-resolution semantics (order, validity, sequencing), because the public
rules describe it that way."""
import jax.numpy as jnp
import pytest

from generals.core import game
from generals.core.env import GeneralsEnv
from generals.core.game import create_initial_state
from generals.modifiers import deathtouch as dt

PASS = jnp.array([1, 0, 0, 0, 0], dtype=jnp.int32)


def make_state(size=6, time=800):
    """Open board, P0 general at (0,0), P1 general at (0, size-1)."""
    grid = jnp.zeros((size, size), dtype=jnp.int32)
    grid = grid.at[0, 0].set(1).at[0, size - 1].set(2)
    state = create_initial_state(grid)
    return state._replace(time=jnp.int32(time))


def give(state, player, ij, army):
    i, j = ij
    return state._replace(
        armies=state.armies.at[i, j].set(army),
        ownership=state.ownership.at[player, i, j].set(True),
        ownership_neutral=state.ownership_neutral.at[i, j].set(False),
    )


def move(i, j, d):  # full-stack move
    return jnp.array([0, i, j, d, 0], dtype=jnp.int32)


RIGHT, LEFT, DOWN, UP = 3, 2, 1, 0


def test_touch_wins_against_a_stacked_general():
    # 5 army touching a 50-army general: hopeless combat, but it's turn 800.
    s = make_state(time=800)
    s = give(s, 0, (0, 4), 5)
    s = s._replace(armies=s.armies.at[0, 5].set(50))
    ns, info = dt.step(s, jnp.stack([move(0, 4, RIGHT), PASS]))
    assert int(info.winner) == 0 and bool(info.is_done)
    # spoils transferred like a normal win
    assert bool(ns.ownership[0, 0, 5])


def test_no_touch_before_the_threshold():
    s = make_state(time=799)
    s = give(s, 0, (0, 4), 5)
    s = s._replace(armies=s.armies.at[0, 5].set(50))
    actions = jnp.stack([move(0, 4, RIGHT), PASS])
    ns, info = dt.step(s, actions)
    assert int(info.winner) == -1 and not bool(info.is_done)
    # bit-identical to the base game: ordinary (losing) combat, plus growth
    base_state, _ = game.step(s, actions)
    assert (ns.armies == base_state.armies).all()
    assert (ns.ownership == base_state.ownership).all()


def test_normal_capture_still_wins_after_threshold():
    s = make_state(time=900)
    s = give(s, 0, (0, 4), 100)
    s = s._replace(armies=s.armies.at[0, 5].set(3))
    _, info = dt.step(s, jnp.stack([move(0, 4, RIGHT), PASS]))
    assert int(info.winner) == 0 and bool(info.is_done)


def test_defense_kills_the_source_no_touch():
    # P0 attacks the general from (0,4); P1 counter-moves (1,4)->(0,4) onto the
    # source. The counter is a chase, so it goes FIRST; 10 > 5 captures the
    # source, so the touch never executes.
    s = make_state(time=800)
    s = give(s, 0, (0, 4), 5)
    s = give(s, 1, (1, 4), 10)
    ns, info = dt.step(s, jnp.stack([move(0, 4, RIGHT), move(1, 4, UP)]))
    assert int(info.winner) == -1 and not bool(info.is_done)
    assert bool(ns.ownership[1, 0, 4])  # defender holds the old source


def test_defense_leaves_survivors_touch_lands():
    # Same chase, but 3 < 5: the source survives with 2 army, one unit still
    # moves — the touch executes and the attacker wins.
    s = make_state(time=800)
    s = give(s, 0, (0, 4), 5)
    s = give(s, 1, (1, 4), 3)
    _, info = dt.step(s, jnp.stack([move(0, 4, RIGHT), move(1, 4, UP)]))
    assert int(info.winner) == 0 and bool(info.is_done)


def test_defense_leaves_exactly_one_no_touch():
    # Defender's 5-stack sends 4 (all-but-one) onto the 5-army source:
    # |5-4| = 1 left — nothing can move, no touch.
    s = make_state(time=800)
    s = give(s, 0, (0, 4), 5)
    s = give(s, 1, (1, 4), 5)
    ns, info = dt.step(s, jnp.stack([move(0, 4, RIGHT), move(1, 4, UP)]))
    assert int(info.winner) == -1 and not bool(info.is_done)
    assert int(ns.armies[0, 4]) == 1


def test_mutual_touch_is_a_draw():
    s = make_state(time=800)
    s = give(s, 0, (0, 4), 5)   # adjacent to P1 general (0,5)
    s = give(s, 1, (0, 1), 5)   # adjacent to P0 general (0,0)
    _, info = dt.step(s, jnp.stack([move(0, 4, RIGHT), move(0, 1, LEFT)]))
    assert bool(info.is_done) and int(info.winner) == -1


def test_invalid_moves_never_touch():
    s = make_state(time=900)
    s = give(s, 0, (0, 4), 1)  # only the garrison — can't move
    _, info = dt.step(s, jnp.stack([move(0, 4, RIGHT), PASS]))
    assert int(info.winner) == -1
    # not your cell
    s2 = make_state(time=900)
    s2 = give(s2, 1, (0, 4), 9)
    _, info2 = dt.step(s2, jnp.stack([move(0, 4, RIGHT), PASS]))
    assert int(info2.winner) == -1


def test_below_threshold_matches_base_step_exactly():
    s = make_state(time=100)
    s = give(s, 0, (2, 2), 12)
    s = give(s, 1, (3, 3), 7)
    actions = jnp.stack([move(2, 2, DOWN), move(3, 3, LEFT)])
    base_state, base_info = game.step(s, actions)
    dt_state, dt_info = dt.step(s, actions)
    assert (base_state.armies == dt_state.armies).all()
    assert (base_state.ownership == dt_state.ownership).all()
    assert int(base_info.winner) == int(dt_info.winner)


def test_env_composition_build_next_to_general_is_not_a_touch():
    # env applies build_castles first (builds become passes), then deathtouch:
    # a build on the tile NEXT to the enemy general must not read as a move.
    env = GeneralsEnv(grid_dims=(6, 6), build_castles=True, deathtouch_turn=800,
                      pool_size=2, truncation=2000)
    import jax.random as jrandom
    pool, _ = env.reset(jrandom.PRNGKey(0))
    s = make_state(time=900)
    s = give(s, 0, (0, 4), 60)  # enough to afford any castle
    build = jnp.array([2, 0, 4, 0, 0], dtype=jnp.int32)
    ts, ns = env.step(s, jnp.stack([build, PASS]), pool)
    assert not bool(ts.terminated)
    assert bool(ns.castles[0, 4])  # the build itself landed


def test_competition_preset_pins_the_format():
    env = GeneralsEnv(mode="competition")
    assert env.truncation == 1200
    assert env.build_castles is True
    assert env.deathtouch_turn == 800
    assert env.perfect_info is False  # fog of war, like the original generals.io
    # rectangular maps: sides drawn in [18, 21], pool padded to 22
    assert env._fixed_dims is None
    assert (env.min_grid_size, env.max_grid_size, env.pad_to) == (18, 21, 22)
