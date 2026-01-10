"""
Simple example of running a single Generals.io game.

This demonstrates the basic game loop with two agents playing against each other.
"""
import jax.numpy as jnp
import jax.random as jrandom

from generals import GeneralsEnv, get_observation
from generals.agents import RandomAgent, ExpanderAgent


# =============================================================================
# Configuration - adjust these to your needs
# =============================================================================
GRID_DIMS = (10, 10)    # Grid size (height, width) - try (15, 15) for larger games
TRUNCATION = 500        # Max steps before game ends

# Create environment
env = GeneralsEnv(grid_dims=GRID_DIMS, truncation=TRUNCATION)

# Create agents
agent_0 = RandomAgent(id="Random")
agent_1 = ExpanderAgent(id="Expander")

# Initialize random key
key = jrandom.PRNGKey(42)
state = env.reset(key)

step_count = 0
terminated = truncated = False

while not (terminated or truncated):
    # Get observations for both players
    obs_0 = get_observation(state, 0)
    obs_1 = get_observation(state, 1)

    key, k1, k2 = jrandom.split(key, 3)
    actions = jnp.stack([
        agent_0.act(obs_0, k1),
        agent_1.act(obs_1, k2),
    ])

    key, step_key = jrandom.split(key)
    timestep, state = env.step(state, actions, step_key)

    terminated = bool(timestep.terminated)
    truncated = bool(timestep.truncated)
    step_count += 1

winner = timestep.info.winner
print(f"Game over! Winner: {winner}")
