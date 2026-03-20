"""
Benchmark: GeneralsEnv throughput on 24x24 grid.

Measures:
  1. Pool generation (upfront cost + memory)
  2. Single-env step throughput
  3. Vectorized throughput (512 envs) via vmap + lax.scan
"""
import jax
import jax.numpy as jnp
import jax.random as jrandom
import time

from generals import GeneralsEnv, get_observation
from generals.agents import RandomAgent
from generals.core.game import step as game_step

GRID = (24, 24)
POOL_SIZE = 10_000


def bench(name, fn, reps=5):
    """Run fn `reps` times, return total steps/sec."""
    # warmup
    result = fn()
    jax.block_until_ready(jax.tree.leaves(result))

    t0 = time.perf_counter()
    for _ in range(reps):
        result = fn()
    jax.block_until_ready(jax.tree.leaves(result))
    elapsed = time.perf_counter() - t0
    return elapsed, reps


# =====================================================================
print("=" * 64)
print(f"  GeneralsEnv Benchmark — {GRID[0]}x{GRID[1]} grid, pool_size={POOL_SIZE}")
print("=" * 64)

# =====================================================================
# 1. Pool generation
# =====================================================================
print(f"\n{'Pool generation':=^64}")
env = GeneralsEnv(grid_dims=GRID, truncation=500, pool_size=POOL_SIZE)

t0 = time.perf_counter()
key = jrandom.PRNGKey(0)
state = env.reset(key)
jax.block_until_ready(jax.tree.leaves(env._pool))
pool_time = time.perf_counter() - t0

pool_bytes = sum(x.nbytes for x in jax.tree.leaves(env._pool))
pool_mb = pool_bytes / 1024 / 1024
per_state = pool_bytes / POOL_SIZE

print(f"  {POOL_SIZE:,} states generated in {pool_time:.2f}s")
print(f"  Memory: {pool_mb:.1f} MB total, {per_state:.0f} bytes/state")

# =====================================================================
# 2. Single-env: full loop (obs + random act + step)
# =====================================================================
print(f"\n{'Single environment':=^64}")

agent = RandomAgent()
STEPS_SINGLE = 500

@jax.jit
def single_step(state, actions):
    return env.step(state, actions)

# Full loop with observation + random agent
key = jrandom.PRNGKey(1)
state = env.init_state(key)

# warmup
for _ in range(20):
    obs0 = get_observation(state, 0)
    obs1 = get_observation(state, 1)
    key, k1, k2 = jrandom.split(key, 3)
    actions = jnp.stack([agent.act(obs0, k1), agent.act(obs1, k2)])
    ts, state = single_step(state, actions)
jax.block_until_ready(state.armies)

key = jrandom.PRNGKey(2)
state = env.init_state(key)
t0 = time.perf_counter()
for _ in range(STEPS_SINGLE):
    obs0 = get_observation(state, 0)
    obs1 = get_observation(state, 1)
    key, k1, k2 = jrandom.split(key, 3)
    actions = jnp.stack([agent.act(obs0, k1), agent.act(obs1, k2)])
    ts, state = single_step(state, actions)
jax.block_until_ready(state.armies)
elapsed = time.perf_counter() - t0
print(f"  Python loop (obs + agent + step): {STEPS_SINGLE / elapsed:>10,.0f} steps/sec")

# Step-only (pass actions, no obs)
dummy_action = jnp.array([[1,0,0,0,0],[1,0,0,0,0]], dtype=jnp.int32)

key = jrandom.PRNGKey(3)
state = env.init_state(key)
for _ in range(20):
    ts, state = single_step(state, dummy_action)
jax.block_until_ready(state.armies)

key = jrandom.PRNGKey(4)
state = env.init_state(key)
t0 = time.perf_counter()
for _ in range(STEPS_SINGLE):
    ts, state = single_step(state, dummy_action)
jax.block_until_ready(state.armies)
elapsed = time.perf_counter() - t0
print(f"  Python loop (step only, pass):    {STEPS_SINGLE / elapsed:>10,.0f} steps/sec")

# =====================================================================
# 3. Vectorized: vmap + lax.scan
# =====================================================================
N_ENVS = 512
N_SCAN = 200
REPS = 5

print(f"\n{'Vectorized (' + str(N_ENVS) + ' envs, lax.scan)':=^64}")

actions_batch = jnp.tile(dummy_action, (N_ENVS, 1, 1))
init_keys = jrandom.split(jrandom.PRNGKey(10), N_ENVS)
init_states = jax.vmap(env.init_state)(init_keys)

# A) env.step via vmap + scan (pool auto-reset)
@jax.jit
def scan_env_step(states):
    def body(states, _):
        ts, new_states = jax.vmap(env.step)(states, actions_batch)
        return new_states, None
    final, _ = jax.lax.scan(body, states, None, length=N_SCAN)
    return final

elapsed, reps = bench("env.step (pool auto-reset)", lambda: scan_env_step(init_states), reps=REPS)
total = N_ENVS * N_SCAN * reps
print(f"  env.step (pool auto-reset):       {total / elapsed:>10,.0f} steps/sec")

# B) raw game_step via vmap + scan (no reset, ceiling)
@jax.jit
def scan_raw_step(states):
    def body(states, _):
        new_states, _ = jax.vmap(game_step)(states, actions_batch)
        return new_states, None
    final, _ = jax.lax.scan(body, states, None, length=N_SCAN)
    return final

elapsed, reps = bench("game_step (no reset, ceiling)", lambda: scan_raw_step(init_states), reps=REPS)
total = N_ENVS * N_SCAN * reps
print(f"  game_step (no reset, ceiling):    {total / elapsed:>10,.0f} steps/sec")

# =====================================================================
# Summary
# =====================================================================
print(f"\n{'Summary':=^64}")
print(f"  Grid:       {GRID[0]}x{GRID[1]}")
print(f"  Pool:       {POOL_SIZE:,} states, {pool_mb:.1f} MB, generated in {pool_time:.1f}s")
print(f"  Vectorized: {N_ENVS} envs x {N_SCAN} steps per scan call")
print()
