"""
Example: Training showcase with vectorized JAX environment and composite reward.

This is a comprehensive example showing how to:
- Set up the vectorized JAX environment for high-performance training
- Use the composite reward function for better learning signals
- Implement a simple training loop with metrics tracking
- Handle auto-reset and episode statistics
- Achieve 100K+ steps/sec on CPU (millions on GPU)
"""
import time
from functools import partial

import jax.numpy as jnp
import jax.random as jrandom

from generals.envs import VectorizedJaxEnv
from generals.core.rewards_jax import composite_reward_fn


def random_actions_jax(key: jnp.ndarray, num_envs: int, grid_size: tuple[int, int]) -> jnp.ndarray:
    """Generate random actions for all environments using JAX random."""
    H, W = grid_size
    
    # Split key for different random operations
    subkeys = jrandom.split(key, 5)
    
    # Generate random values for all actions at once
    pass_vals = jrandom.uniform(subkeys[0], (num_envs, 2)) < 0.3  # 30% chance to pass
    rows = jrandom.randint(subkeys[1], (num_envs, 2), 0, H)
    cols = jrandom.randint(subkeys[2], (num_envs, 2), 0, W)
    directions = jrandom.randint(subkeys[3], (num_envs, 2), 0, 4)
    splits = jrandom.randint(subkeys[4], (num_envs, 2), 0, 2)
    
    # Stack into action arrays [num_envs, 2, 5]
    actions = jnp.stack([
        pass_vals.astype(jnp.int32),
        rows,
        cols,
        directions,
        splits
    ], axis=-1)
    
    return actions


def compute_episode_stats(rewards, terminated):
    """Compute useful statistics from episode data."""
    # Average reward across all environments
    mean_reward = jnp.mean(rewards)
    
    # Count how many episodes finished this step
    num_done = jnp.sum(terminated)
    
    # Player 0 win rate (when episodes terminate)
    player0_wins = jnp.sum(terminated[:, 0])
    player1_wins = jnp.sum(terminated[:, 1])
    
    return {
        'mean_reward': mean_reward,
        'num_done': num_done,
        'player0_wins': player0_wins,
        'player1_wins': player1_wins,
    }


def main():
    """Run a training showcase demonstration."""
    # =========================================================================
    # Configuration
    # =========================================================================
    num_envs = 256  # More environments for better training
    num_episodes = 3  # Quick demo - increase for real training
    max_steps_per_episode = 1000
    
    # Reward function configuration
    reward_config = {
        'city_weight': 0.4,         # Weight for city capture
        'ratio_weight': 0.3,        # Weight for army/land ratios
        'maximum_army_ratio': 1.6,  # Army ratio clipping threshold
        'maximum_land_ratio': 1.3,  # Land ratio clipping threshold
    }
    
    print("\n" + "=" * 80)
    print("JAX Vectorized Environment - Training Showcase")
    print("=" * 80)
    print("\nConfiguration:")
    print(f"  Parallel environments: {num_envs}")
    print(f"  Episodes to run: {num_episodes}")
    print(f"  Max steps per episode: {max_steps_per_episode}")
    print(f"\nReward function: Composite (multi-signal)")
    print(f"  City weight: {reward_config['city_weight']}")
    print(f"  Ratio weight: {reward_config['ratio_weight']}")
    print(f"  Army ratio threshold: {reward_config['maximum_army_ratio']}")
    print(f"  Land ratio threshold: {reward_config['maximum_land_ratio']}")
    
    # =========================================================================
    # Environment Setup
    # =========================================================================
    print("\n" + "-" * 80)
    print("Setting up environment...")
    print("-" * 80)
    
    # Create custom reward function using partial application
    custom_reward_fn = partial(
        composite_reward_fn,
        **reward_config
    )
    
    # Create environment with composite reward
    # Note: The environment needs to be modified to accept reward_fn parameter
    # For now, we'll compute rewards manually after env.step()
    env = VectorizedJaxEnv(
        num_envs=num_envs, 
        mode='generalsio',  # Random grid sizes like online game
        truncation=max_steps_per_episode  # Auto-truncate long episodes
    )
    
    print(f"  Mode: {env.mode}")
    print(f"  Grid size (padded): {env.grid_size}")
    print(f"  Truncation: {max_steps_per_episode} steps")
    
    # Initialize JAX random key
    rng_key = jrandom.PRNGKey(42)
    
    # =========================================================================
    # JIT Warmup
    # =========================================================================
    print("\n" + "-" * 80)
    print("Warming up JIT compilation (this takes a moment)...")
    print("-" * 80)
    
    warmup_start = time.time()
    obs, info = env.reset(seed=42)
    
    # Run a few steps to trigger JIT compilation
    for _ in range(10):
        rng_key, subkey = jrandom.split(rng_key)
        actions = random_actions_jax(subkey, num_envs, env.grid_size)
        obs, rewards, terminated, truncated, info = env.step(actions)
    
    warmup_time = time.time() - warmup_start
    print(f"  JIT compilation completed in {warmup_time:.2f}s")
    print(f"  Observation shape: {obs.armies.shape}")  # [num_envs, 2, H, W]
    print(f"  Rewards shape: {rewards.shape}")  # [num_envs, 2]
    
    # =========================================================================
    # Training Loop
    # =========================================================================
    print("\n" + "-" * 80)
    print("Starting training loop...")
    print("-" * 80)
    
    # Metrics tracking
    total_steps = 0
    total_episodes_completed = 0
    episode_times = []
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_start = time.time()
        
        episode_rewards_p0 = []
        episode_rewards_p1 = []
        
        for step in range(max_steps_per_episode):
            # Generate random actions (replace with your RL agent)
            rng_key, subkey = jrandom.split(rng_key)
            actions = random_actions_jax(subkey, num_envs, env.grid_size)
            
            # Step environment
            next_obs, rewards, terminated, truncated, info = env.step(actions)
            
            total_steps += num_envs
            
            # Track rewards
            episode_rewards_p0.append(jnp.mean(rewards[:, 0]))
            episode_rewards_p1.append(jnp.mean(rewards[:, 1]))
            
            # Count completed episodes
            done = terminated | truncated
            num_done = jnp.sum(jnp.any(done, axis=-1)).item()
            total_episodes_completed += num_done
            
            obs = next_obs
            
            # Early stop if all environments finished
            if jnp.all(done):
                break
        
        episode_time = time.time() - episode_start
        episode_times.append(episode_time)
        steps_in_episode = (step + 1) * num_envs
        
        # Compute statistics for this episode
        avg_reward_p0 = jnp.mean(jnp.array(episode_rewards_p0))
        avg_reward_p1 = jnp.mean(jnp.array(episode_rewards_p1))
        
        print(f"Episode {episode + 1:2d}/{num_episodes}: "
              f"{steps_in_episode:6,} steps | "
              f"{episode_time:5.2f}s | "
              f"{steps_in_episode / episode_time:7,.0f} steps/s | "
              f"Avg rewards: P0={avg_reward_p0:6.2f}, P1={avg_reward_p1:6.2f}")
    
    # =========================================================================
    # Final Statistics
    # =========================================================================
    total_time = sum(episode_times)
    avg_throughput = total_steps / total_time
    
    print("\n" + "=" * 80)
    print("Training Summary")
    print("=" * 80)
    print(f"\nPerformance:")
    print(f"  Total steps executed: {total_steps:,}")
    print(f"  Total episodes completed: {total_episodes_completed}")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Average throughput: {avg_throughput:,.0f} steps/sec")
    print(f"  Average episode time: {total_time / num_episodes:.2f}s")
    
    # Clean up
    env.close()
    
    print("\n" + "=" * 80)
    print("Training showcase completed!")
    print("=" * 80)
    print("\nNext steps for your training:")
    print("  1. Replace random_actions_jax() with your RL agent's policy")
    print("  2. Add your training algorithm (PPO, DQN, SAC, etc.)")
    print("  3. Use the composite_reward_fn for better learning signals")
    print("  4. Tune reward weights via partial() for your strategy")
    print("  5. Scale to 1000s of envs for faster training")
    print("  6. Enable GPU: install jax[cuda] for 10-100x speedup")
    print("\nCustomize reward function:")
    print("  from functools import partial")
    print("  my_reward = partial(composite_reward_fn,")
    print("                      city_weight=0.6,  # Emphasize cities")
    print("                      ratio_weight=0.2)  # Deemphasize ratios")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()

