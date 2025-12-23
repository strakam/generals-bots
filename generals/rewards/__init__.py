"""Reward functions for Generals.io environments."""

from generals.rewards.jax_rewards import (
    REWARD_FUNCTIONS,
    get_reward_fn,
    land_difference,
    army_and_land,
    win_lose,
    land_with_win_bonus,
)

__all__ = [
    'REWARD_FUNCTIONS',
    'get_reward_fn',
    'land_difference',
    'army_and_land',
    'win_lose',
    'land_with_win_bonus',
]
