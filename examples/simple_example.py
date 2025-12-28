import jax.numpy as jnp
import jax.random as jrandom
from generals.core.env import GeneralsEnv
from generals.core.game import get_observation


# Initialize environment
env = GeneralsEnv(truncation=100)
# Initialize random key
key = jrandom.PRNGKey(42)

# Reset environment
state = env.reset(key)

terminated = truncated = False
step_count = 0

while not (terminated or truncated) and step_count < 20:
    # Get observations for both players
    obs_p0 = get_observation(state, 0)
    obs_p1 = get_observation(state, 1)
    
    # Create dummy actions for both players (both pass)
    actions = jnp.stack([
        jnp.array([1, 0, 0, 0, 0], dtype=jnp.int32),
        jnp.array([1, 0, 0, 0, 0], dtype=jnp.int32)
    ])
    
    # Step environment
    timestep, state = env.step(state, actions, key)
    
    # Extract results
    obs = timestep.observation
    rewards = timestep.reward
    terminated = bool(timestep.terminated)
    truncated = bool(timestep.truncated)
    info = timestep.info

    step_count += 1
    
    # Split key for next iteration
    key, _ = jrandom.split(key)

print("\nExample complete!")
print(f"Final state - Time: {int(state.time)}, Winner: {int(state.winner)}")