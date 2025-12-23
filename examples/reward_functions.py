"""
Example: Using different reward functions with VectorizedJaxEnv

This example shows how to configure different reward functions
for training Generals.io agents.
"""

import jax.numpy as jnp
from generals.envs.jax_env import VectorizedJaxEnv

# 1. Use built-in reward by name (default is 'land_difference')
env = VectorizedJaxEnv(
    num_envs=128,
    reward_fn='land_difference'  # Focus on territorial control
)

# 2. Try a different built-in reward
env = VectorizedJaxEnv(
    num_envs=128,
    reward_fn='win_lose'  # Sparse reward: only +1/-1 at end
)

# 3. Combine land and army in reward
env = VectorizedJaxEnv(
    num_envs=128,
    reward_fn='army_and_land'  # 0.3*army_diff + 0.7*land_diff
)

# 4. Use land shaping with win bonus
env = VectorizedJaxEnv(
    num_envs=128,
    reward_fn='land_with_win_bonus'  # Continuous + sparse terminal
)

# 5. Add truncation for faster training
env = VectorizedJaxEnv(
    num_envs=128,
    reward_fn='land_difference',
    truncation=500  # Episode ends after 500 steps
)

# 6. Define custom reward function
def aggressive_reward(info, state):
    """Reward aggressive play: emphasize army count"""
    army_diff = info.army[0] - info.army[1]
    land_diff = info.land[0] - info.land[1]
    
    # 70% army, 30% land
    combined = 0.7 * army_diff + 0.3 * land_diff
    return jnp.array([combined, -combined])

env = VectorizedJaxEnv(
    num_envs=128,
    reward_fn=aggressive_reward,  # Pass the function directly
    truncation=1000  # Optional: limit episode length
)

# The reward function signature is:
# (info: GameInfo, state: GameState) -> Array[2]
# Returns rewards for [player_0, player_1] for a single environment.
# The environment automatically vmaps this over batches.

# Available built-in rewards (see generals/rewards/jax_rewards.py):
# - 'land_difference': Territory-focused (default)
# - 'army_and_land': Balanced army and territory
# - 'win_lose': Sparse terminal rewards only
# - 'land_with_win_bonus': Shaping + terminal bonus

# Truncation vs Termination:
# - Terminated: Game ended naturally (someone won)
# - Truncated: Hit max timesteps (truncation parameter)
# - Terminated has priority if both occur simultaneously
# - Both trigger auto-reset in vectorized environment

print("Reward functions configured!")
