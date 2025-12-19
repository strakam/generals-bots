"""
Simple Gym-like vectorized environment using JAX.

This example shows how to use the JAX implementation with a standard
reinforcement learning loop pattern (reset, step, done handling).
"""
import time
from typing import Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from generals.core import game_jax
from generals.core.grid import GridFactory


class VectorizedJaxEnv:
    """
    Vectorized environment wrapper for JAX game implementation.
    Provides a simple Gym-like interface for batched environments.
    """
    
    def __init__(self, num_envs: int, grid_size: Tuple[int, int] = (10, 10)):
        """
        Args:
            num_envs: Number of parallel environments
            grid_size: Grid dimensions (height, width)
        """
        self.num_envs = num_envs
        self.grid_size = grid_size
        
        # Create grid factory
        self.grid_factory = GridFactory(
            min_grid_dims=grid_size,
            max_grid_dims=grid_size,
            mountain_density=0.1,
            city_density=0.1,
            general_positions=[[1, 1], [grid_size[0]-2, grid_size[1]-2]],
        )
        
        # JIT compile the step function for performance
        self._jitted_step = jax.jit(game_jax.batch_step)
        self._jitted_single_step = jax.jit(game_jax.step)
        
        # Initialize states
        self.states = None
        
    def reset(self) -> Dict[str, jnp.ndarray]:
        """
        Reset all environments.
        
        Returns:
            Batched observations dict with shape [num_envs, ...]
        """
        # Create initial state (same grid for all envs for simplicity)
        grid = self.grid_factory.generate()
        grid_array = np.vectorize(ord)(grid.grid).astype(np.uint8)
        grid_jax = jnp.array(grid_array)
        
        single_state = game_jax.create_initial_state(grid_jax)
        
        # Batch it
        self.states = jax.tree.map(
            lambda x: jnp.stack([x] * self.num_envs),
            single_state
        )
        
        # Get initial observations
        obs = self._get_observations()
        
        return obs
    
    def step(self, actions: jnp.ndarray) -> Tuple[Dict[str, jnp.ndarray], jnp.ndarray, jnp.ndarray, Dict]:
        """
        Step all environments with given actions.
        
        Args:
            actions: [num_envs, 2, 5] array of actions for both players
        
        Returns:
            observations: Batched observations dict
            rewards: [num_envs, 2] rewards for both players
            dones: [num_envs] boolean array indicating if episodes are done
            infos: Dict with additional info
        """
        # Step environments
        self.states, infos = self._jitted_step(self.states, actions)
        
        # Get observations
        obs = self._get_observations()
        
        # Compute rewards (simple: land difference)
        rewards = infos['land'][:, 0] - infos['land'][:, 1]  # Player 0 perspective
        rewards = jnp.stack([rewards, -rewards], axis=1)  # Both players
        
        # Check if done
        dones = infos['is_done']
        
        # Convert infos to regular dict
        info_dict = {
            'army': np.array(infos['army']),
            'land': np.array(infos['land']),
            'winner': np.array(infos['winner']),
            'time': np.array(infos['time']),
        }
        
        return obs, rewards, dones, info_dict
    
    def _get_observations(self) -> Dict[str, jnp.ndarray]:
        """
        Get observations for all environments and both players.
        
        Returns:
            Dict with batched observations [num_envs, 2, ...] for each field
        """
        # JIT compiled observation extraction
        if not hasattr(self, '_jitted_get_obs'):
            @jax.jit
            def get_both_obs(state):
                obs_p0 = game_jax.get_observation(state, 0)
                obs_p1 = game_jax.get_observation(state, 1)
                
                # Stack observations for both players
                stacked = {}
                for key in obs_p0.keys():
                    stacked[key] = jnp.stack([obs_p0[key], obs_p1[key]])
                return stacked
            
            # Vectorize over batch
            self._jitted_get_obs = jax.jit(jax.vmap(get_both_obs))
        
        return self._jitted_get_obs(self.states)


def random_actions(num_envs: int, grid_size: Tuple[int, int]) -> jnp.ndarray:
    """Generate random actions for all environments."""
    actions = []
    for _ in range(num_envs):
        env_actions = []
        for _ in range(2):  # 2 players
            if np.random.random() < 0.3:  # 30% chance to pass
                action = [1, 0, 0, 0, 0]
            else:
                action = [
                    0,  # don't pass
                    np.random.randint(0, grid_size[0]),  # row
                    np.random.randint(0, grid_size[1]),  # col
                    np.random.randint(0, 4),  # direction
                    np.random.randint(0, 2),  # split
                ]
            env_actions.append(action)
        actions.append(env_actions)
    
    return jnp.array(actions, dtype=jnp.int32)


def main():
    """Run a simple training loop demonstration."""
    print("=" * 70)
    print("Vectorized JAX Environment Demo")
    print("=" * 70)
    
    # Configuration
    num_envs = 64
    grid_size = (20, 20)
    num_episodes = 5
    max_steps_per_episode = 2000
    
    print(f"\nConfiguration:")
    print(f"  Number of parallel environments: {num_envs}")
    print(f"  Grid size: {grid_size}")
    print(f"  Episodes to run: {num_episodes}")
    print(f"  Max steps per episode: {max_steps_per_episode}")
    
    # Create environment
    print("\nCreating environment...")
    env = VectorizedJaxEnv(num_envs=num_envs, grid_size=grid_size)
    
    # Reset and warmup JIT
    print("Warming up JIT compilation...")
    obs = env.reset()
    for _ in range(5):
        actions = random_actions(num_envs, grid_size)
        obs, rewards, dones, info = env.step(actions)
    
    print("\nStarting training loop...\n")
    
    total_steps = 0
    episode_rewards = []
    
    for episode in range(num_episodes):
        obs = env.reset()
        episode_reward = np.zeros((num_envs, 2))
        episode_length = np.zeros(num_envs)
        
        episode_start = time.time()
        
        for step in range(max_steps_per_episode):
            # Generate random actions (replace with policy in real training)
            actions = random_actions(num_envs, grid_size)
            
            # Step environment
            obs, rewards, dones, info = env.step(actions)
            
            # Accumulate rewards
            episode_reward += np.array(rewards)
            episode_length += ~dones
            
            total_steps += num_envs
            
            # Print progress every 50 steps
            if step % 50 == 0:
                num_done = np.sum(dones)
                avg_army_p0 = np.mean(info['army'][:, 0])
                avg_army_p1 = np.mean(info['army'][:, 1])
                avg_land_p0 = np.mean(info['land'][:, 0])
                avg_land_p1 = np.mean(info['land'][:, 1])
                
                print(f"  Episode {episode+1}, Step {step:3d}: "
                      f"{num_done:2d} done | "
                      f"Army: P0={avg_army_p0:6.1f} P1={avg_army_p1:6.1f} | "
                      f"Land: P0={avg_land_p0:5.1f} P1={avg_land_p1:5.1f}")
            
            # Check if all done
            if np.all(dones):
                break
        
        episode_time = time.time() - episode_start
        
        # Episode summary
        avg_reward_p0 = np.mean(episode_reward[:, 0])
        avg_reward_p1 = np.mean(episode_reward[:, 1])
        avg_length = np.mean(episode_length)
        wins_p0 = np.sum(info['winner'] == 0)
        wins_p1 = np.sum(info['winner'] == 1)
        
        print(f"\n  Episode {episode+1} finished:")
        print(f"    Time: {episode_time:.2f}s")
        print(f"    Steps: {int(avg_length)} (avg)")
        print(f"    Rewards: P0={avg_reward_p0:.1f}, P1={avg_reward_p1:.1f} (avg)")
        print(f"    Wins: P0={wins_p0}/{num_envs}, P1={wins_p1}/{num_envs}")
        print(f"    Throughput: {num_envs * int(avg_length) / episode_time:.0f} steps/sec")
        print()
        
        episode_rewards.append(avg_reward_p0)
    
    # Final summary
    print("=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"\nTotal steps: {total_steps:,}")
    print(f"Average reward (P0): {np.mean(episode_rewards):.1f}")
    
    # Demonstrate observation structure
    print("\n" + "=" * 70)
    print("Observation Structure")
    print("=" * 70)
    
    obs = env.reset()
    print(f"\nObservation dict keys: {list(obs.keys())}")
    print(f"\nShapes:")
    for key, value in obs.items():
        print(f"  {key:25s}: {value.shape}")
    
    print(f"\nExample values (env 0, player 0):")
    print(f"  Owned land count: {obs['owned_land_count'][0, 0]}")
    print(f"  Owned army count: {obs['owned_army_count'][0, 0]}")
    print(f"  Opponent land count: {obs['opponent_land_count'][0, 0]}")
    print(f"  Opponent army count: {obs['opponent_army_count'][0, 0]}")
    print(f"  Timestep: {obs['timestep'][0, 0]}")
    
    # Show a small part of the spatial arrays
    print(f"\n  Armies (first 5x5):")
    print(f"    {obs['armies'][0, 0, :5, :5]}")
    
    print(f"\n  Owned cells (first 5x5):")
    print(f"    {obs['owned_cells'][0, 0, :5, :5].astype(int)}")


if __name__ == "__main__":
    main()
