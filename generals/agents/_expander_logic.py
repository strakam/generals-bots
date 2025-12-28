"""JAX-compatible expander agent logic (internal module)."""
import jax
import jax.numpy as jnp
import jax.random as jrandom

from generals.core.action import compute_valid_move_mask_obs, DIRECTIONS


@jax.jit
def expander_action(key: jnp.ndarray, observation) -> jnp.ndarray:
    """
    Expander agent that prioritizes border expansion with strongest cells.

    Strategy:
    1. Find all cells on the border (owned cells adjacent to non-owned)
    2. Score moves by: army_count * can_capture * (opponent_bonus)
    3. Prefer moving strongest border cells to capture new territory
    """
    valid_mask = compute_valid_move_mask_obs(observation)
    H, W = observation.armies.shape

    valid_positions = jnp.argwhere(valid_mask, size=H * W * 4, fill_value=-1)
    num_valid = jnp.sum(jnp.all(valid_positions >= 0, axis=-1))
    should_pass = num_valid == 0

    def evaluate_move(idx):
        move = valid_positions[idx]
        is_valid = jnp.all(move >= 0)

        orig_row, orig_col, direction = move[0], move[1], move[2]

        di, dj = DIRECTIONS[direction]
        dest_row = jnp.clip(orig_row + di, 0, H - 1)
        dest_col = jnp.clip(orig_col + dj, 0, W - 1)

        orig_armies = observation.armies[orig_row, orig_col]
        dest_armies = observation.armies[dest_row, dest_col]
        is_opponent = observation.opponent_cells[dest_row, dest_col]
        is_neutral = observation.neutral_cells[dest_row, dest_col]
        is_owned = observation.owned_cells[dest_row, dest_col]

        can_capture = orig_armies > dest_armies + 1
        is_expansion = ~is_owned & (is_opponent | is_neutral)

        score = orig_armies.astype(jnp.float32)
        score = jnp.where(is_expansion & can_capture, score * 10.0, score)
        opponent_multiplier = jnp.where(is_opponent, 2.0, 1.0)
        score = jnp.where(is_expansion & can_capture, score * opponent_multiplier, score)
        score = jnp.where(is_valid & can_capture, score, 0.0)

        return score

    scores = jax.vmap(evaluate_move)(jnp.arange(H * W * 4))
    has_expansion_captures = jnp.sum(scores) > 0

    probs = jnp.where(
        has_expansion_captures,
        scores,
        jnp.where(jnp.arange(H * W * 4) < num_valid, jnp.ones(H * W * 4), jnp.zeros(H * W * 4)),
    )
    probs = probs / (jnp.sum(probs) + 1e-8)

    move_idx = jrandom.choice(key, H * W * 4, p=probs)
    selected_move = valid_positions[move_idx]

    return jnp.array(
        [should_pass.astype(jnp.int32), selected_move[0], selected_move[1], selected_move[2], jnp.int32(0)],
        dtype=jnp.int32,
    )

