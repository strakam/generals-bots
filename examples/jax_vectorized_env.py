"""
Simple Gym-like vectorized environment using JAX.

This example shows how to use the JAX implementation with a standard
reinforcement learning loop pattern (reset, step, done handling).
Heavily optimized with JIT compilation and JAX random for maximum performance.
"""
import time
from typing import Dict, Tuple

import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np

from generals.core import game_jax
from generals.core.grid import GridFactory
from generals.core.observation_jax import ObservationJax


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
        
    def reset(self) -> ObservationJax:
        """
        Reset all environments.
        
        Returns:
            Batched ObservationJax with shape [num_envs, 2, ...]
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
    
    def step(self, actions: jnp.ndarray) -> Tuple[ObservationJax, jnp.ndarray, jnp.ndarray, Dict]:
        """
        Step all environments with given actions.
        
        Args:
            actions: [num_envs, 2, 5] array of actions for both players
        
        Returns:
            observations: Batched ObservationJax NamedTuple
            rewards: [num_envs, 2] rewards for both players
            dones: [num_envs] boolean array indicating if episodes are done
            infos: Dict with additional info
        """
        # Step environments
        self.states, infos = self._jitted_step(self.states, actions)
        
        # Get observations
        obs = self._get_observations()
        
        # Compute rewards (simple: land difference)
        rewards = infos.land[:, 0] - infos.land[:, 1]  # Player 0 perspective
        rewards = jnp.stack([rewards, -rewards], axis=1)  # Both players
        
        # Check if done
        dones = infos.is_done
        
        # Convert infos to regular dict (for compatibility)
        info_dict = {
            'army': np.array(infos.army),
            'land': np.array(infos.land),
            'winner': np.array(infos.winner),
            'time': np.array(infos.time),
        }
        
        return obs, rewards, dones, info_dict
    
    def _get_observations(self) -> ObservationJax:
        """
        Get observations for all environments and both players.
        
        Returns:
            Batched ObservationJax with shape [num_envs, 2, ...] for each field
        """
        # JIT compiled observation extraction
        if not hasattr(self, '_jitted_get_obs'):
            @jax.jit
            def get_both_obs(state):
                obs_p0 = game_jax.get_observation(state, 0)
                obs_p1 = game_jax.get_observation(state, 1)
                
                # Stack observations for both players as NamedTuple
                return jax.tree.map(
                    lambda x, y: jnp.stack([x, y]),
                    obs_p0,
                    obs_p1
                )
            
            # Vectorize over batch
            self._jitted_get_obs = jax.jit(jax.vmap(get_both_obs))
        
        return self._jitted_get_obs(self.states)


def random_actions_jax(key: jnp.ndarray, num_envs: int, grid_size: Tuple[int, int]) -> jnp.ndarray:
    """Generate random actions for all environments using JAX random (vectorized)."""
    H, W = grid_size
    
    # Split key for different random operations
    subkeys = jrandom.split(key, 5)
    
    # Generate random values for all actions at once
    pass_vals = jrandom.uniform(subkeys[0], (num_envs, 2)) < 0.3  # 30% chance to pass
    rows = jrandom.randint(subkeys[1], (num_envs, 2), 0, H)
    cols = jrandom.randint(subkeys[2], (num_envs, 2), 0, W)
    directions = jrandom.randint(subkeys[3], (num_envs, 2), 0, 4)
    splits = jrandom.randint(subkeys[4], (num_envs, 2), 0, 2)
    
    # Stack into action arrays
    actions = jnp.stack([
        pass_vals.astype(jnp.int32),
        rows,
        cols,
        directions,
        splits
    ], axis=-1)
    
    return actions


def main():
    """Run a simple training loop demonstration."""
    # Configuration
    num_envs = 128
    grid_size = (20, 20)
    num_episodes = 5
    max_steps_per_episode = 2000
    
    print(f"\nConfiguration:")
    print(f"  Number of parallel environments: {num_envs}")
    print(f"  Grid size: {grid_size}")
    print(f"  Episodes to run: {num_episodes}")
    print(f"  Max steps per episode: {max_steps_per_episode}")
    
    # Create environment
    env = VectorizedJaxEnv(num_envs=num_envs, grid_size=grid_size)
    
    # Initialize JAX random key
    rng_key = jrandom.PRNGKey(42)
    
    # Reset and warmup JIT
    obs = env.reset()
    print(f"\nObservation type: {type(obs).__name__}")
    print(f"Observation fields: {obs._fields}")
    print(f"Example field shape (armies): {obs.armies.shape}")
    
    print("\nWarming up JIT compilation...")
    for _ in range(5):
        rng_key, subkey = jrandom.split(rng_key)
        actions = random_actions_jax(subkey, num_envs, grid_size)
        obs, rewards, dones, info = env.step(actions)
    
    total_steps = 0
    times = []
    
    print("\nRunning episodes...")
    for episode in range(num_episodes):
        obs = env.reset()
        episode_reward = jnp.zeros((num_envs, 2))
        episode_length = jnp.zeros(num_envs)
        
        episode_start = time.time()
        
        for step in range(max_steps_per_episode):
            # Generate random actions using JAX random (FAST!)
            rng_key, subkey = jrandom.split(rng_key)
            actions = random_actions_jax(subkey, num_envs, grid_size)
            
            # Step environment
            obs, rewards, dones, info = env.step(actions)
            
            # Accumulate rewards
            episode_reward += rewards
            episode_length += ~dones
            total_steps += num_envs
            
            # Check if all done
            if jnp.all(dones):
                break
        
        episode_time = time.time() - episode_start
        times.append(episode_time)
    
    print(f"\nPerformance:")
    print(f"  Total steps: {total_steps:,}")
    print(f"  Time: {sum(times):.2f}s")
    print(f"  Throughput: {total_steps / sum(times):,.0f} steps/sec")


if __name__ == "__main__":
    main()
