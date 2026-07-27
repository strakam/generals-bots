"""Move-order tiebreak: chasing > reinforcing > SMALLER army.

Smaller-first is deliberate. On a contested cell the bigger force resolves
last and ends up holding it — larger-first let the smaller force snipe a
neutral garrison the bigger one had just paid for. And a deathtouch head-on
clash (both moves targeting each other's source) goes to the attacker, so
past the threshold the endgame stays a forced finish: the defense against a
touch is a chase from a third tile, not the general counter-launching.
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


def test_bigger_army_ends_up_holding_a_contested_neutral_cell():
    """Both race for a neutral garrison: the smaller force enters first and
    pays it down, the bigger force resolves last and takes the cell. Under
    larger-first this came out backwards — the sniper won."""
    s = board()
    s = give(s, 0, (2, 1), 25)
    s = give(s, 1, (2, 3), 40)
    s = s._replace(armies=s.armies.at[2, 2].set(20))  # neutral garrison
    a0 = jnp.array([0, 2, 1, RIGHT, 0], dtype=jnp.int32)  # 25 → (2,2)
    a1 = jnp.array([0, 2, 3, LEFT, 0], dtype=jnp.int32)   # 40 → (2,2)
    s2, _ = game.step(s, jnp.stack([a0, a1]))
    assert bool(s2.ownership[1, 2, 2]), "the bigger army must end up holding the cell"
    # P0's 24 clears the 20 garrison and holds 4; P1's 39 then takes it: 39-4.
    assert int(s2.armies[2, 2]) == 35


def test_deathtouch_head_on_goes_to_the_attacker():
    """P1 touches from (0,1) while P0's general counter-launches onto (0,1).
    Mutual chase, so the tie falls to the smaller army — the attacker moves
    first and the touch wins, no matter how big the general's stack is."""
    s = board(time=900)
    s = s._replace(armies=s.armies.at[0, 0].set(50))
    s = give(s, 1, (0, 1), 10)
    a0 = jnp.array([0, 0, 0, RIGHT, 0], dtype=jnp.int32)  # general 50 → (0,1)
    a1 = jnp.array([0, 0, 1, LEFT, 0], dtype=jnp.int32)   # attacker 10 → (0,0)
    _, info = dt.step(s, jnp.stack([a0, a1]), COMPETITION_DEATHTOUCH_TURN)
    assert bool(info.is_done)
    assert int(info.winner) == 1


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
