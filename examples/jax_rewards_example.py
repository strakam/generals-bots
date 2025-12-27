"""
Example: Using different reward functions with JAX

This example demonstrates how to use the various reward functions
available in the JAX implementation.
"""

import jax.numpy as jnp
from functools import partial

from generals.core.rewards_jax import (
    composite_reward_fn,
    city_reward_fn,
    ratio_reward_fn,
    win_lose_reward_fn,
)

# Assuming you have prior_obs, prior_action, and obs from your JAX environment
# prior_obs: ObservationJax at time t-1
# prior_action: jnp.ndarray of shape [5]
# obs: ObservationJax at time t

print("=== Method 1: Direct function calls (simplest) ===")

# Use composite reward with default parameters
reward = composite_reward_fn(prior_obs, prior_action, obs)

# Use city-focused reward
reward = city_reward_fn(prior_obs, prior_action, obs)

# Use ratio-focused reward
reward = ratio_reward_fn(prior_obs, prior_action, obs)

# Use sparse win/lose reward
reward = win_lose_reward_fn(prior_obs, prior_action, obs)


print("\n=== Method 2: Partial application for custom hyperparameters ===")

# Create a custom composite reward with different weights
my_composite_reward = partial(
    composite_reward_fn,
    city_weight=0.5,           # More weight on cities
    ratio_weight=0.4,          # More weight on ratios
    maximum_army_ratio=2.0,    # Higher army ratio threshold
    maximum_land_ratio=1.5     # Higher land ratio threshold
)

# Now use it like a regular function
reward = my_composite_reward(prior_obs, prior_action, obs)


# Create an aggressive ratio-based reward
aggressive_reward = partial(
    ratio_reward_fn,
    clip_value=2.0,       # Higher clip value for more variance
    shaping_weight=0.8    # Strong shaping signal
)

reward = aggressive_reward(prior_obs, prior_action, obs)


print("\n=== Recommended: Use partial for training loops ===")

# For training loops where you call the reward many times:
# Use partial application for best JAX performance
reward_fn = partial(composite_reward_fn, city_weight=0.5, ratio_weight=0.3)

# In your training loop:
# for step in range(num_steps):
#     ...
#     reward = reward_fn(prior_obs, prior_action, obs)
#     ...


print("\n=== Summary of available rewards ===")
print("""
1. composite_reward_fn (recommended)
   - Combines army ratio, land ratio, cities, and win/lose
   - Most comprehensive, good for general training
   - Hyperparameters: city_weight, ratio_weight, maximum_army_ratio, maximum_land_ratio
   - Defaults: city_weight=0.4, ratio_weight=0.3, maximum_army_ratio=1.6, maximum_land_ratio=1.3

2. city_reward_fn
   - Focuses on capturing cities
   - Good for emphasizing strategic positions
   - Hyperparameters: shaping_weight
   - Default: shaping_weight=0.3

3. ratio_reward_fn
   - Based on army ratio between players
   - Good for teaching relative strength awareness
   - Hyperparameters: clip_value, shaping_weight
   - Defaults: clip_value=1.5, shaping_weight=0.5

4. win_lose_reward_fn
   - Sparse reward: only at game end (+1 win, -1 lose)
   - Small bonus for split actions (+0.0015)
   - No hyperparameters
""")

print("\nAll reward functions are @jax.jit compiled for performance!")

