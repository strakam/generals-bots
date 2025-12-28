"""Example of running multiple parallel environments with JAX vmap."""
import jax
import jax.numpy as jnp
import jax.random as jrandom
import time

from generals import GeneralsEnv, get_observation
from generals.agents import RandomAgent, ExpanderAgent

# Create vectorized environment with 1024 parallel games
num_envs = 256
env = GeneralsEnv(num_envs=num_envs, truncation=500)

agent_0 = RandomAgent()
agent_1 = ExpanderAgent()

# Initialize with different seeds for each environment
key = jrandom.PRNGKey(42)
keys = jrandom.split(key, num_envs)
states = env.reset(keys)

print(f"Running {num_envs} parallel environments...")
print(f"State shape: armies={states.armies.shape}, ownership={states.ownership.shape}")
print("-" * 60)

start_time = time.time()
total_steps = 0

# Run for 100 steps (batched across all environments)
for step in range(100):
    # Get observations for all environments
    # We vmap get_observation over the batch dimension
    get_obs_p0 = jax.vmap(lambda s: get_observation(s, 0))
    get_obs_p1 = jax.vmap(lambda s: get_observation(s, 1))
    
    obs_p0 = get_obs_p0(states)  # Batched observations for player 0
    obs_p1 = get_obs_p1(states)  # Batched observations for player 1
    
    # Get actions from agents (vmap over environments)
    key, *subkeys = jrandom.split(key, num_envs * 2 + 1)
    keys_p0 = jnp.array(subkeys[:num_envs])
    keys_p1 = jnp.array(subkeys[num_envs:])
    
    actions_p0 = jax.vmap(agent_0.act)(obs_p0, keys_p0)
    actions_p1 = jax.vmap(agent_1.act)(obs_p1, keys_p1)
    actions = jnp.stack([actions_p0, actions_p1], axis=1)  # Shape: [num_envs, 2, 5]
    
    # Step all environments in parallel
    key, *step_keys = jrandom.split(key, num_envs + 1)
    step_keys = jnp.array(step_keys)
    timesteps, states = env.step(states, actions, step_keys)
    
    total_steps += num_envs
    
    if step % 20 == 0:
        # Compute statistics
        avg_p0_land = jnp.mean(timesteps.info.land[:, 0])
        avg_p1_land = jnp.mean(timesteps.info.land[:, 1])
        num_done = jnp.sum(timesteps.terminated | timesteps.truncated)
        
        elapsed = time.time() - start_time
        steps_per_sec = total_steps / elapsed if elapsed > 0 else 0
        
        print(f"Step {step:3d}: P0 land={avg_p0_land:5.1f}, P1 land={avg_p1_land:5.1f}, "
              f"done={int(num_done):4d}/{num_envs}, {steps_per_sec:,.0f} steps/s")

elapsed = time.time() - start_time
steps_per_sec = total_steps / elapsed

print("-" * 60)
print(f"Total time: {elapsed:.2f}s")
print(f"Throughput: {steps_per_sec:,.0f} steps/second")
print(f"Per environment: {100 / elapsed:.1f} steps/second")

