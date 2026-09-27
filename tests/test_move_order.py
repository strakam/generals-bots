"""Move order is generals.io's: defensive moves first, moves onto a general
tile last (within each class, a friendly merge onto a general included), then
the LARGER army first, equal armies in player order (reversed on odd turns),
and a chased piece waits for its chaser. The engine replicates the
website game rather than a rule of its own; the consequences on a contested
cell and for the deathtouch modifier are pinned here.
"""
import jax.numpy as jnp

from generals.core import game
from generals.modifiers import deathtouch as dt

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


def test_larger_army_enters_a_contested_neutral_cell_first():
    """Both race for a neutral garrison: the larger force enters first and pays
    it down, the smaller force arrives second and can take the cell from what
    is left (generals.io's rule)."""
    s = board()
    s = give(s, 0, (2, 1), 25)
    s = give(s, 1, (2, 3), 40)
    s = s._replace(armies=s.armies.at[2, 2].set(20))  # neutral garrison
    a0 = jnp.array([0, 2, 1, RIGHT, 0], dtype=jnp.int32)  # 25 → (2,2)
    a1 = jnp.array([0, 2, 3, LEFT, 0], dtype=jnp.int32)   # 40 → (2,2)
    s2, _ = game.step(s, jnp.stack([a0, a1]))
    assert bool(s2.ownership[0, 2, 2]), "the larger army resolves first; the smaller one takes what is left"
    # P1's 39 clears the 20 garrison and holds 19; P0's 24 then takes it: 24-19.
    assert int(s2.armies[2, 2]) == 5


def test_deathtouch_head_on_the_general_counter_launch_resolves_first():
    """P1 touches from (0,1) while P0's general counter-launches onto (0,1).
    A touch is an attack on a general, which generals.io's order resolves last,
    so the counter-launch goes first: it strips the attacker's source and the
    touch never executes."""
    s = board(time=900)
    s = s._replace(armies=s.armies.at[0, 0].set(50))
    s = give(s, 1, (0, 1), 10)
    a0 = jnp.array([0, 0, 0, RIGHT, 0], dtype=jnp.int32)  # general 50 → (0,1)
    a1 = jnp.array([0, 0, 1, LEFT, 0], dtype=jnp.int32)   # attacker 10 → (0,0)
    s2, info = dt.step(s, jnp.stack([a0, a1]), COMPETITION_DEATHTOUCH_TURN)
    assert not bool(info.is_done), "the counter-launch resolves before the touch and strips its source"
    assert bool(s2.ownership[0, 0, 1]) and int(s2.armies[0, 1]) == 39


def test_deathtouch_head_on_touch_executes_when_the_counter_launch_is_too_small():
    """Same head-on, but the general's launch cannot clear the source: at least
    one unit still stands there, the touch executes second, attacker wins."""
    s = board(time=900)
    s = s._replace(armies=s.armies.at[0, 0].set(5))
    s = give(s, 1, (0, 1), 10)
    a0 = jnp.array([0, 0, 0, RIGHT, 0], dtype=jnp.int32)  # general 5 → (0,1): 4 vs 10, fails
    a1 = jnp.array([0, 0, 1, LEFT, 0], dtype=jnp.int32)   # attacker 10 → (0,0)
    _, info = dt.step(s, jnp.stack([a0, a1]), COMPETITION_DEATHTOUCH_TURN)
    assert bool(info.is_done) and int(info.winner) == 1


def test_deathtouch_chase_from_a_third_tile_still_defends():
    """The real defense: a chase from a tile that is not the general. Chase
    priority beats the army tiebreak, the source is captured, and the touch
    never executes."""
    s = board(time=900)
    s = s._replace(armies=s.armies.at[0, 0].set(50))
    s = give(s, 1, (0, 1), 10)
    s = give(s, 0, (1, 1), 20)
    a0 = jnp.array([0, 1, 1, UP, 0], dtype=jnp.int32)     # 20 chases (0,1)
    a1 = jnp.array([0, 0, 1, LEFT, 0], dtype=jnp.int32)   # touch attempt → (0,0)
    s2, info = dt.step(s, jnp.stack([a0, a1]), COMPETITION_DEATHTOUCH_TURN)
    assert not bool(info.is_done), "the chase must strip the touch at its source"
    assert bool(s2.ownership[0, 0, 1]), "the chaser captures the attacker's source"


def test_a_merge_onto_a_teammates_general_sorts_last_among_defensive_moves():
    """generals.io's isGeneralAttack flags every move whose destination is a
    general tile, defensive ones included. Head-on swap between teammates in
    2v2: P0 splits 10 from his general onto P1's tile, P1 moves 30 from that
    tile onto the general. Both moves are defensive; P1's is onto a general,
    so P0's resolves first despite the smaller army, and P1's then finds its
    source gone (larger-army-first would give (0,0)=20, (0,1)=20)."""
    grid = jnp.zeros((6, 6), dtype=jnp.int32).at[0, 0].set(1).at[0, 5].set(2).at[5, 0].set(3).at[5, 5].set(4)
    s = game.create_initial_state(grid, teams=jnp.array([0, 0, 1, 1], dtype=jnp.int32))
    s = s._replace(armies=s.armies.at[0, 0].set(10))
    s = give(s, 1, (0, 1), 30)
    a0 = jnp.array([0, 0, 0, RIGHT, 1], dtype=jnp.int32)   # P0 general -> (0,1), split: 5 move
    a1 = jnp.array([0, 0, 1, LEFT, 0], dtype=jnp.int32)    # P1 (0,1) -> P0's general
    p = jnp.array([1, 0, 0, 0, 0], dtype=jnp.int32)
    order = game._determine_move_order(s, jnp.stack([a0, a1, p, p]))
    assert order.tolist()[:2] == [0, 1]
    s2, _ = game.step(s, jnp.stack([a0, a1, p, p]))
    assert int(s2.armies[0, 0]) == 5 and bool(s2.ownership[0, 0, 0])
    assert int(s2.armies[0, 1]) == 35 and bool(s2.ownership[0, 0, 1])
