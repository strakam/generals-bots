"""
JAX-compatible Expander Agent.

Strategy: Move cells with the most armies that are on the border of owned territory
to capture new cells (opponent or neutral). This creates aggressive outward expansion.
"""
import jax
import jax.numpy as jnp
import jax.random as jrandom

from generals.core.action import compute_valid_move_mask_obs, DIRECTIONS


@jax.jit
def expander_agent_jax(key: jnp.ndarray, observation) -> jnp.ndarray:
    """
    Expander agent that prioritizes border expansion with strongest cells.
    
    Strategy:
    1. Find all cells on the border (owned cells adjacent to non-owned)
    2. Score moves by: army_count * can_capture * (opponent_bonus)
    3. Prefer moving strongest border cells to capture new territory
    
    Args:
        key: JAX random key
        observation: Observation NamedTuple
    
    Returns:
        Action array [5]: [pass, row, col, direction, split]
    """
    # Compute valid moves [H, W, 4]
    valid_mask = compute_valid_move_mask_obs(observation)
    
    H, W = observation.armies.shape
    
    # Get all valid positions
    valid_positions = jnp.argwhere(valid_mask, size=H*W*4, fill_value=-1)
    num_valid = jnp.sum(jnp.all(valid_positions >= 0, axis=-1))
    
    # If no valid moves, pass
    should_pass = num_valid == 0
    
    def evaluate_move(idx):
        """
        Evaluate a single move based on:
        - Army count at source (prefer stronger cells)
        - Whether it captures new territory
        - Whether it's an opponent cell (bonus)
        """
        move = valid_positions[idx]
        is_valid = jnp.all(move >= 0)
        
        orig_row, orig_col, direction = move[0], move[1], move[2]
        
        # Get destination
        di, dj = DIRECTIONS[direction]
        dest_row = orig_row + di
        dest_col = orig_col + dj
        
        # Clamp to valid range
        dest_row = jnp.clip(dest_row, 0, H - 1)
        dest_col = jnp.clip(dest_col, 0, W - 1)
        
        # Source cell info
        orig_armies = observation.armies[orig_row, orig_col]
        
        # Destination cell info
        dest_armies = observation.armies[dest_row, dest_col]
        is_opponent = observation.opponent_cells[dest_row, dest_col]
        is_neutral = observation.neutral_cells[dest_row, dest_col]
        is_owned = observation.owned_cells[dest_row, dest_col]
        
        # Can we capture this cell?
        can_capture = orig_armies > dest_armies + 1
        
        # Is this an expansion move (to non-owned territory)?
        is_expansion = ~is_owned & (is_opponent | is_neutral)
        
        # Calculate score
        # Base score: number of armies (prefer moving strong cells)
        score = orig_armies.astype(jnp.float32)
        
        # Multiply by 10 if it captures new territory
        score = jnp.where(is_expansion & can_capture, score * 10.0, score)
        
        # Bonus multiplier for capturing opponent (2x) vs neutral (1x)
        opponent_multiplier = jnp.where(is_opponent, 2.0, 1.0)
        score = jnp.where(is_expansion & can_capture, score * opponent_multiplier, score)
        
        # Zero out invalid moves or moves that don't capture
        score = jnp.where(is_valid & can_capture, score, 0.0)
        
        return score
    
    # Vectorize evaluation over all potential moves
    scores = jax.vmap(evaluate_move)(jnp.arange(H*W*4))
    
    # Create selection probabilities
    # If there are any expansion captures, focus on those
    # Otherwise, allow any valid move (including reinforcing borders)
    has_expansion_captures = jnp.sum(scores) > 0
    
    probs = jnp.where(
        has_expansion_captures,
        scores,
        # Fallback: any valid move with equal probability
        jnp.where(
            jnp.arange(H*W*4) < num_valid,
            jnp.ones(H*W*4),
            jnp.zeros(H*W*4)
        )
    )
    
    # Normalize probabilities
    probs = probs / (jnp.sum(probs) + 1e-8)
    
    # Sample move (weighted by score, so stronger cells more likely to be chosen)
    move_idx = jrandom.choice(key, H*W*4, p=probs)
    selected_move = valid_positions[move_idx]
    
    # Build action
    action = jnp.array([
        should_pass.astype(jnp.int32),
        selected_move[0],
        selected_move[1],
        selected_move[2],
        jnp.int32(0),  # Don't split
    ], dtype=jnp.int32)
    
    return action
