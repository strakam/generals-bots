"""generals.io tunnels: a tile that admits at most `limit` armies per move; the rest stays behind."""
import jax.numpy as jnp
import numpy as np

from generals.core import game


def _grid():
    g = np.zeros((4, 4), dtype=np.int32)
    g[0, 0] = 1          # P0 general
    g[3, 3] = 2          # P1 general
    return g


def _state(limits=None):
    s = game.create_initial_state(jnp.asarray(_grid()), tunnel_limits=limits)
    armies = s.armies.at[0, 1].set(45)                      # P0 stack next to the tunnel at (0, 2)
    ownership = s.ownership.at[0, 0, 1].set(True)
    return s._replace(armies=armies, ownership=ownership,
                      ownership_neutral=s.ownership_neutral.at[0, 1].set(False))


def _move_right(s):
    acts = jnp.array([[0, 0, 1, 3, 0], [1, 0, 0, 0, 0]], dtype=jnp.int32)      # P0: (0,1) -> right; P1 pass
    d = int(jnp.argmax(jnp.all(game.DIRECTIONS == jnp.array([0, 1]), axis=1)))
    acts = acts.at[0, 3].set(d)
    return game.step(s, acts)[0]


def test_no_tunnel_moves_all_but_one():
    s = _move_right(_state())
    assert int(s.armies[0, 1]) == 1 and int(s.armies[0, 2]) == 44


def test_tunnel_caps_incoming_army():
    limits = np.zeros((4, 4), dtype=np.int32); limits[0, 2] = 25
    s = _move_right(_state(limits))
    assert int(s.armies[0, 1]) == 20 and int(s.armies[0, 2]) == 25      # generals.io replay 3Cm16hPth, turn 188
    assert bool(s.ownership[0, 0, 2])


def test_tunnel_cap_applies_to_attacks_too():
    limits = np.zeros((4, 4), dtype=np.int32); limits[0, 2] = 25
    s = _state(limits)
    s = s._replace(armies=s.armies.at[0, 2].set(30), ownership=s.ownership.at[1, 0, 2].set(True),
                   ownership_neutral=s.ownership_neutral.at[0, 2].set(False))
    s = _move_right(s)
    assert int(s.armies[0, 1]) == 20 and int(s.armies[0, 2]) == 5 and bool(s.ownership[1, 0, 2])


def test_default_state_has_no_tunnels():
    tl = game.create_initial_state(jnp.asarray(_grid())).tunnel_limits
    assert tl.shape == (4, 4) and int(tl.sum()) == 0


def test_tunnel_overflow_is_refunded_to_the_source():
    # generals.io replay 3Cm16hPth, turn 188: 45 armies move into a limit-25 tunnel already holding 1;
    # 25 enter, the tile ends at 26 > 25, and the excess 1 is refunded to the source (21 / 25).
    limits = np.zeros((4, 4), dtype=np.int32); limits[0, 2] = 25
    s = _state(limits)
    s = s._replace(armies=s.armies.at[0, 2].set(1), ownership=s.ownership.at[0, 0, 2].set(True),
                   ownership_neutral=s.ownership_neutral.at[0, 2].set(False))
    s = _move_right(s)
    assert int(s.armies[0, 1]) == 21 and int(s.armies[0, 2]) == 25
