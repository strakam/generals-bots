"""Deathtouch modifier: from a configured turn, a move that EXECUTES onto the
enemy general's tile wins instantly — army counts are irrelevant.

The touch condition deliberately reuses the base engine's own semantics
instead of inventing new ones:

  - Moves resolve one after the other in game.step's order (chasing >
    reinforcing > larger army — see game._determine_move_order). A "touch" is
    a move that is VALID at its own slot in that order (same validity test as
    game._execute_move) and whose destination is the opponent's general tile.
  - That gives the defense its teeth: a counter-move onto the attacker's
    SOURCE tile is a chase, so it goes first. Capture the source outright and
    the touch never executes (its source is no longer the attacker's).
    Leave the source with 2+ army — at least one unit still moves — and the
    touch executes anyway: attacker wins.
  - Both players touching on the same turn is a DRAW: reported like a
    truncation draw (info.is_done with winner -1).
  - A normal general capture at/after the threshold is itself a touch, so the
    two win paths always agree.

Composition: builds (generals.modifiers.build_castles) are rewritten to
passes BEFORE the step, so a build next to the enemy general never counts as
a touch — only pass-field 0 (a real move) can.
"""
import jax
import jax.numpy as jnp
from jax import lax

from generals.core import game

DEATHTOUCH_TURN = 800  # default threshold; GeneralsEnv passes its own


def _executes_onto_general(state: game.GameState, player_idx, action) -> jnp.ndarray:
    """True iff this action, applied to THIS state, is a valid move whose
    destination is the opponent's general tile. Mirrors game._execute_move's
    validity test exactly — if that changes, change this."""
    pass_turn, si, sj, direction, split_army = action
    H, W = state.armies.shape

    in_bounds = (si >= 0) & (si < H) & (sj >= 0) & (sj < W)
    di = si + game.DIRECTIONS[direction, 0]
    dj = sj + game.DIRECTIONS[direction, 1]
    dest_in_bounds = (di >= 0) & (di < H) & (dj >= 0) & (dj < W)

    owns_source = state.ownership[player_idx, si, sj]
    source_army = state.armies[si, sj]
    army_to_move = lax.cond(split_army == 1, lambda a: a // 2, lambda a: a - 1, source_army)
    army_to_move = jnp.maximum(0, jnp.minimum(army_to_move, source_army - 1))
    valid = in_bounds & dest_in_bounds & owns_source & (army_to_move > 0) & state.passable[di, dj]

    g = state.general_positions[1 - player_idx]
    return (pass_turn == 0) & valid & (di == g[0]) & (dj == g[1])


@jax.jit
def step(state: game.GameState, actions: jnp.ndarray,
         turn: int = DEATHTOUCH_TURN) -> tuple[game.GameState, game.GameInfo]:
    """Drop-in replacement for game.step with the deathtouch rule active.

    The touch test replays the exact order the base step is about to use:
    evaluate the first mover's touch against the pre-step state, execute that
    action, then evaluate the second mover's touch against the updated state.
    game.step then runs unmodified (its own execution recomputes the same
    deterministic order), and the outcome is overridden only when a touch
    actually happened.
    """
    active = (state.winner < 0) & (state.time >= turn)

    first = game._determine_move_order(state, actions)
    t_first = _executes_onto_general(state, first, actions[first])
    mid = game.execute_action(state, first, actions[first])
    t_second = _executes_onto_general(mid, 1 - first, actions[1 - first])
    # map (first, second) back to (player 0, player 1)
    touch = jnp.where(first == 0,
                      jnp.stack([t_first, t_second]),
                      jnp.stack([t_second, t_first])) & active

    new_state, _ = game.step(state, actions)

    both = touch[0] & touch[1]
    one = touch[0] ^ touch[1]
    toucher = jnp.where(touch[0], 0, 1).astype(new_state.winner.dtype)

    # A lone touch can only ever agree with a winner the base step already set
    # (any base-game capture at/after the threshold is itself a touch; a win
    # by the NON-toucher would require capturing a general, i.e. touching).
    winner = jnp.where(both, jnp.int32(-1), jnp.where(one, toucher, new_state.winner))
    new_state = new_state._replace(winner=winner)
    # Forced win the base step didn't settle: apply the usual spoils.
    need_transfer = one & (winner >= 0) & (state.winner < 0)
    new_state = lax.cond(need_transfer, game._transfer_loser_cells_to_winner,
                         lambda s: s, new_state)

    info = game.get_info(new_state)
    # Mutual touch: the state has no draw encoding (winner stays -1), so the
    # episode ends the way truncation draws do — is_done with winner -1.
    info = info._replace(is_done=info.is_done | both)
    return new_state, info
