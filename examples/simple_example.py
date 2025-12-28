import jax.numpy as jnp
import jax.random as jrandom

from generals import GeneralsEnv, get_observation
from generals.agents import RandomAgent, ExpanderAgent


# Create environment with truncation at 500 steps
env = GeneralsEnv(truncation=500, render=True)

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
