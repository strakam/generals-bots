"""
JAX-based vectorized Generals.io environment.

Provides a high-performance, Gym-compatible vectorized environment for training
RL agents. Uses JAX for maximum throughput (~45k+ steps/sec).

Features:
- Vectorized execution (run many environments in parallel)
- Full JAX implementation (JIT-compiled)
- Auto-reset on episode termination
- Standard Gym API (v0.26+)
- Multi-agent (2 players per environment)
"""

from typing import Dict, Tuple, Optional, Any

import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np

from generals.core import game_jax
from generals.core.grid import GridFactory
from generals.core.observation_jax import ObservationJax
from generals.core.game_jax import GameInfo, GameState


class VectorizedJaxEnv:
    """
    Vectorized environment wrapper for JAX game implementation.
    Provides a standard Gym-like interface for batched environments.
    
    This is a vectorized multi-agent environment where each environment
    has 2 players. Observations and rewards are returned for both players.
    
    Example:
        >>> env = VectorizedJaxEnv(num_envs=128, grid_size=(20, 20))
        >>> obs, info = env.reset(seed=42)
        >>> actions = jnp.zeros((128, 2, 5), dtype=jnp.int32)  # Random actions
        >>> obs, rewards, terminated, truncated, info = env.step(actions)
        >>> env.close()
    """
    
    # Gym metadata
    metadata = {
        'render_modes': ['human', 'rgb_array'],
        'render_fps': 10,
    }
    
    def __init__(
        self, 
        num_envs: int, 
        grid_size: Tuple[int, int] = (10, 10),
        render_mode: Optional[str] = None,
        mountain_density: float = 0.1,
        city_density: float = 0.1,
    ):
        """
        Initialize vectorized JAX environment.
        
        Args:
            num_envs: Number of parallel environments
            grid_size: Grid dimensions (height, width)
            render_mode: Mode for rendering ('human', 'rgb_array', or None)
            mountain_density: Probability of mountain on each cell
            city_density: Probability of city on each cell
        """
        self.num_envs = num_envs
        self.grid_size = grid_size
        self.render_mode = render_mode
        
        # Create grid factory
        self.grid_factory = GridFactory(
            min_grid_dims=grid_size,
            max_grid_dims=grid_size,
            mountain_density=mountain_density,
            city_density=city_density,
            general_positions=[[1, 1], [grid_size[0]-2, grid_size[1]-2]],
        )
        
        # JIT compile the step function for performance
        self._jitted_step = jax.jit(game_jax.batch_step)
        self._jitted_single_step = jax.jit(game_jax.step)
        
        # JIT compile auto-reset helper
        self._jitted_auto_reset = None  # Lazy init after first reset
        
        # Initialize states
        self.states = None
        self._rng_key = jrandom.PRNGKey(0)
        
        # Cache initial state for fast auto-reset
        self._initial_state_template = None
        self._batched_initial_state = None  # Pre-batched for speed
        
        # Define spaces (for Gym compatibility)
        # Note: These are approximate as action/observation spaces are complex
        # For a full Gym wrapper, you'd use gym.spaces
        self.action_space = None  # Placeholder: would be Discrete or MultiDiscrete
        self.observation_space = None  # Placeholder: would be Dict or Box
        
    def reset(
        self, 
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[ObservationJax, GameInfo]:
        """
        Reset all environments.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional options (unused for now)
        
        Returns:
            observations: Batched ObservationJax with shape [num_envs, 2, ...]
            infos: GameInfo NamedTuple with initial game state info
        """
        # Set seed if provided
        if seed is not None:
            self._rng_key = jrandom.PRNGKey(seed)
        
        # Create initial state (same grid for all envs for simplicity)
        grid = self.grid_factory.generate()
        grid_array = np.vectorize(ord)(grid.grid).astype(np.uint8)
        grid_jax = jnp.array(grid_array)
        
        single_state = game_jax.create_initial_state(grid_jax)
        
        # Cache for auto-reset
        self._initial_state_template = single_state
        
        # Pre-batch initial state for fast auto-reset
        self._batched_initial_state = jax.tree.map(
            lambda x: jnp.stack([x] * self.num_envs),
            single_state
        )
        
        # Batch current states
        self.states = self._batched_initial_state
        
        # Get initial observations and info
        obs = self._get_observations()
        
        # Get batched info (vmap get_info over batch dimension)
        infos = jax.vmap(game_jax.get_info)(self.states)
        
        return obs, infos
    
    def step(
        self, 
        actions: jnp.ndarray,
    ) -> Tuple[ObservationJax, jnp.ndarray, jnp.ndarray, jnp.ndarray, GameInfo]:
        """
        Step all environments with given actions.
        Automatically resets any terminated environments.
        
        Args:
            actions: [num_envs, 2, 5] array of actions for both players
        
        Returns:
            observations: Batched ObservationJax NamedTuple
            rewards: [num_envs, 2] rewards for both players (JAX array)
            terminated: [num_envs] boolean array indicating if episodes ended
            truncated: [num_envs] boolean array (always False for now)
            infos: GameInfo NamedTuple with batched game state info
        """
        # Step environments
        new_states, infos = self._jitted_step(self.states, actions)
        
        # Auto-reset: Replace terminated environments with initial state
        # Use JIT-compiled helper for maximum performance
        if self._batched_initial_state is not None:
            # Lazy init of JIT-compiled reset function
            if self._jitted_auto_reset is None:
                @jax.jit
                def auto_reset_states(new_states, init_states, is_done_mask):
                    """JIT-compiled auto-reset: replace done envs with initial state."""
                    def select_state(new_val, init_val):
                        # Broadcast mask to match dimensions
                        mask = is_done_mask
                        while mask.ndim < new_val.ndim:
                            mask = mask[..., None]
                        # where(condition, if_true, if_false)
                        return jnp.where(mask, init_val, new_val)
                    
                    return jax.tree.map(select_state, new_states, init_states)
                
                self._jitted_auto_reset = auto_reset_states
            
            # Apply auto-reset (JIT-compiled, no Python overhead)
            self.states = self._jitted_auto_reset(
                new_states, 
                self._batched_initial_state, 
                infos.is_done
            )
        else:
            self.states = new_states
        
        # Get observations (after potential reset)
        obs = self._get_observations()
        
        # Compute rewards (simple: land difference)
        # Note: rewards are from the TERMINATED episode, before reset
        rewards = infos.land[:, 0] - infos.land[:, 1]  # Player 0 perspective
        rewards = jnp.stack([rewards, -rewards], axis=1)  # Both players
        
        # Check if done
        terminated = infos.is_done
        
        # Truncated (for now always False - could add max_steps in future)
        truncated = jnp.zeros(self.num_envs, dtype=jnp.bool_)
        
        return obs, rewards, terminated, truncated, infos
    
    def render(self) -> Optional[np.ndarray]:
        """
        Render the environment.
        
        Returns:
            RGB array if render_mode='rgb_array', None otherwise
        """
        if self.render_mode == 'rgb_array':
            # TODO: Implement rendering to RGB array
            # For now, return placeholder
            return np.zeros((self.num_envs, *self.grid_size, 3), dtype=np.uint8)
        elif self.render_mode == 'human':
            # TODO: Implement human-visible rendering
            # For now, just print state
            print(f"Env states: {self.num_envs} environments running")
            return None
        return None
    
    def close(self):
        """
        Close the environment and clean up resources.
        """
        # JAX doesn't need explicit cleanup, but we can reset state
        self.states = None
    
    def seed(self, seed: Optional[int] = None):
        """
        Set the random seed for the environment.
        
        Args:
            seed: Random seed
        
        Returns:
            List containing the seed
        """
        if seed is not None:
            self._rng_key = jrandom.PRNGKey(seed)
        return [seed]
    
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
