"""Vectorized environments example - run many games in parallel with jax.vmap."""
import jax
import jax.numpy as jnp
import jax.random as jrandom
import time

from generals import GeneralsEnv, get_observation
from generals.agents import RandomAgent, ExpanderAgent

NUM_ENVS = 256
GRID_DIMS = (10, 10)

# Create environment and agents
env = GeneralsEnv(grid_dims=GRID_DIMS, truncation=500)
agent_0 = RandomAgent()
agent_1 = ExpanderAgent()

# Vectorize functions using jax.vmap
reset_vmap = jax.vmap(env.reset)
step_vmap = jax.vmap(env.step)
get_obs_p0 = jax.vmap(lambda s: get_observation(s, 0))
get_obs_p1 = jax.vmap(lambda s: get_observation(s, 1))
act_p0 = jax.vmap(agent_0.act)
act_p1 = jax.vmap(agent_1.act)

# Initialize environments
key = jrandom.PRNGKey(42)
reset_keys = jrandom.split(key, NUM_ENVS)
states = reset_vmap(reset_keys)

# Game loop
for step_idx in range(300):
    # Get observations and actions
    obs_p0 = get_obs_p0(states)
    obs_p1 = get_obs_p1(states)
    
    key, *subkeys = jrandom.split(key, NUM_ENVS * 2 + 1)
    keys_p0 = jnp.array(subkeys[:NUM_ENVS])
    keys_p1 = jnp.array(subkeys[NUM_ENVS:])
    
    actions_p0 = act_p0(obs_p0, keys_p0)
    actions_p1 = act_p1(obs_p1, keys_p1)
    actions = jnp.stack([actions_p0, actions_p1], axis=1)
    
    # Step all environments
    key, *step_keys = jrandom.split(key, NUM_ENVS + 1)
    timesteps, states = step_vmap(states, actions, jnp.array(step_keys))
    
    if step_idx % 20 == 0:
        avg_p0_land = jnp.mean(timesteps.info.land[:, 0])
        avg_p1_land = jnp.mean(timesteps.info.land[:, 1])
        num_done = jnp.sum(timesteps.terminated | timesteps.truncated)
        print(f"Step {step_idx:3d}: P0={avg_p0_land:5.1f} P1={avg_p1_land:5.1f}")
