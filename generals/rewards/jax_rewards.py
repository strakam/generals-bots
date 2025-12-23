"""
JAX reward functions for Generals.io environment.

Each reward function has signature:
    (info: GameInfo, state: GameState) -> Array[2]
    
Where the returned array contains rewards for [player_0, player_1].
The environment vmaps these functions over batches automatically.
"""

import jax.numpy as jnp
from generals.core.game_jax import GameInfo, GameState


def land_difference(info: GameInfo, state: GameState) -> jnp.ndarray:
    """
    Reward based on land count difference.
    Player gets +1 for each land tile they have more than opponent.
    """
    diff = info.land[0] - info.land[1]
    return jnp.array([diff, -diff])


def army_and_land(info: GameInfo, state: GameState) -> jnp.ndarray:
    """
    Reward combining army count and land count.
    Weighted sum: 0.3 * army_diff + 0.7 * land_diff
    """
    army_diff = info.army[0] - info.army[1]
    land_diff = info.land[0] - info.land[1]
    combined = 0.3 * army_diff + 0.7 * land_diff
    return jnp.array([combined, -combined])


def win_lose(info: GameInfo, state: GameState) -> jnp.ndarray:
    """
    Sparse reward: +1 for win, -1 for loss, 0 otherwise.
    """
    is_p0_winner = (info.is_done & (info.winner == 0)).astype(jnp.float32)
    is_p1_winner = (info.is_done & (info.winner == 1)).astype(jnp.float32)
    
    p0_reward = is_p0_winner - is_p1_winner
    p1_reward = is_p1_winner - is_p0_winner
    
    return jnp.array([p0_reward, p1_reward])


def land_with_win_bonus(info: GameInfo, state: GameState) -> jnp.ndarray:
    """
    Land difference + large bonus for winning.
    Continuous shaping + sparse terminal reward.
    """
    land_diff = info.land[0] - info.land[1]
    
    # Add win bonus
    win_bonus = jnp.where(
        info.is_done & (info.winner == 0),
        100.0,
        jnp.where(info.is_done & (info.winner == 1), -100.0, 0.0)
    )
    
    combined = land_diff + win_bonus
    return jnp.array([combined, -combined])


# Registry for easy lookup
REWARD_FUNCTIONS = {
    'land_difference': land_difference,
    'army_and_land': army_and_land,
    'win_lose': win_lose,
    'land_with_win_bonus': land_with_win_bonus,
}


def get_reward_fn(name_or_fn):
    """
    Get a reward function by name or return the function if already callable.
    
    Args:
        name_or_fn: Either a string key from REWARD_FUNCTIONS or a callable
        
    Returns:
        Reward function with signature (GameInfo, GameState) -> Array[2]
        The environment vmaps this over batches to get Array[num_envs, 2]
    """
    if callable(name_or_fn):
        return name_or_fn
    
    if name_or_fn not in REWARD_FUNCTIONS:
        raise ValueError(
            f"Unknown reward function: {name_or_fn}. "
            f"Available: {list(REWARD_FUNCTIONS.keys())}"
        )
    
    return REWARD_FUNCTIONS[name_or_fn]
