"""
Performance benchmark for vectorized environments.

Measures throughput (steps/second) with different configurations.
Uses jax.lax.scan for maximum performance.

Usage:
    python benchmark_performance.py [num_envs] [num_steps] [iterations]
    
Examples:
    python benchmark_performance.py 256 100 10
    python benchmark_performance.py 1024 500 5
    python benchmark_performance.py 4096 100 3
"""
import sys
import time
import jax
import jax.numpy as jnp
import jax.random as jrandom

from generals import GeneralsEnv, get_observation
from generals.agents import RandomAgent

# Parse arguments
NUM_ENVS = int(sys.argv[1]) if len(sys.argv) > 1 else 256
NUM_STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 100
ITERATIONS = int(sys.argv[3]) if len(sys.argv) > 3 else 5
GRID_DIMS = (10, 10)

print("=" * 70)
print("GENERALS.IO PERFORMANCE BENCHMARK")
print("=" * 70)
print(f"Configuration:")
print(f"  Num environments: {NUM_ENVS:,}")
print(f"  Steps per iter:   {NUM_STEPS:,}")
print(f"  Iterations:       {ITERATIONS}")
print(f"  Grid size:        {GRID_DIMS[0]}x{GRID_DIMS[1]}")
print(f"  Device:           {jax.devices()[0]}")
print(f"  Total steps:      {NUM_ENVS * NUM_STEPS * ITERATIONS:,}")
print("=" * 70)

# Create environment and agents
env = GeneralsEnv(grid_dims=GRID_DIMS, truncation=500)
agent_0 = RandomAgent()
agent_1 = RandomAgent()


def make_step_fn(env, agent_0, agent_1, num_envs):
    """Create a single vectorized step function."""
    
    step_vmap = jax.vmap(env.step)
    get_obs_p0 = jax.vmap(lambda s: get_observation(s, 0))
    get_obs_p1 = jax.vmap(lambda s: get_observation(s, 1))
    act_p0 = jax.vmap(agent_0.act)
    act_p1 = jax.vmap(agent_1.act)
    
    def step_fn(carry, _):
        states, key = carry
        
        # Get observations
        obs_p0 = get_obs_p0(states)
        obs_p1 = get_obs_p1(states)
        
        # Get actions
        key, k1, k2 = jrandom.split(key, 3)
        keys_p0 = jrandom.split(k1, num_envs)
        keys_p1 = jrandom.split(k2, num_envs)
        
        actions_p0 = act_p0(obs_p0, keys_p0)
        actions_p1 = act_p1(obs_p1, keys_p1)
        actions = jnp.stack([actions_p0, actions_p1], axis=1)
        
        # Step environments
        key, step_key = jrandom.split(key)
        step_keys = jrandom.split(step_key, num_envs)
        timesteps, new_states = step_vmap(states, actions, step_keys)
        
        # Count done episodes
        done_count = jnp.sum(timesteps.terminated | timesteps.truncated)
        
        return (new_states, key), done_count
    
    return step_fn


def make_rollout_fn(env, agent_0, agent_1, num_envs, num_steps):
    """Create a jitted rollout function using lax.scan."""
    
    step_fn = make_step_fn(env, agent_0, agent_1, num_envs)
    
    @jax.jit
    def rollout(states, key):
        (final_states, final_key), done_counts = jax.lax.scan(
            step_fn,
            (states, key),
            None,
            length=num_steps
        )
        total_done = jnp.sum(done_counts)
        return final_states, final_key, total_done
    
    return rollout


# Create rollout function
rollout_fn = make_rollout_fn(env, agent_0, agent_1, NUM_ENVS, NUM_STEPS)

# Initialize environments
key = jrandom.PRNGKey(42)
reset_keys = jrandom.split(key, NUM_ENVS)
reset_vmap = jax.vmap(env.reset)
states = reset_vmap(reset_keys)

print("\nWarming up JIT compilation...")
key, subkey = jrandom.split(key)
states, key, _ = rollout_fn(states, subkey)
jax.block_until_ready(states)
print("Warmup complete!\n")

# Benchmark
print("Running benchmark...")
print("-" * 70)

iteration_times = []
all_stats = []

for iteration in range(ITERATIONS):
    key, subkey = jrandom.split(key)
    
    iter_start = time.time()
    states, key, episode_count = rollout_fn(states, subkey)
    jax.block_until_ready(states)
    iter_elapsed = time.time() - iter_start
    
    iteration_times.append(iter_elapsed)
    sps = (NUM_ENVS * NUM_STEPS) / iter_elapsed
    all_stats.append({
        'sps': sps,
        'episodes': int(episode_count),
        'time': iter_elapsed
    })
    
    print(f"Iteration {iteration + 1}/{ITERATIONS}: "
          f"{sps:>10,.0f} steps/s | "
          f"{iter_elapsed:>6.3f}s | "
          f"{int(episode_count):>4} episodes")

print("-" * 70)

# Summary statistics
avg_sps = sum(s['sps'] for s in all_stats) / len(all_stats)
max_sps = max(s['sps'] for s in all_stats)
min_sps = min(s['sps'] for s in all_stats)
total_time = sum(iteration_times)
total_steps = NUM_ENVS * NUM_STEPS * ITERATIONS
total_episodes = sum(s['episodes'] for s in all_stats)

print("\nBENCHMARK RESULTS")
print("=" * 70)
print(f"Total time:          {total_time:.2f}s")
print(f"Total steps:         {total_steps:,}")
print(f"Total episodes:      {total_episodes:,}")
print(f"")
print(f"Average throughput:  {avg_sps:,.0f} steps/s")
print(f"Peak throughput:     {max_sps:,.0f} steps/s")
print(f"Min throughput:      {min_sps:,.0f} steps/s")
print(f"")
print(f"Per-env throughput:  {avg_sps / NUM_ENVS:.1f} steps/s")
print(f"Steps/episode:       {total_steps / max(total_episodes, 1):.1f}")
print("=" * 70)
