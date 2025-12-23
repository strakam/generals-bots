"""
JAX-based vectorized Generals.io environment.

Pure JAX implementation optimized for GPU training.
Uses functional grid generation for fast, vectorized operations.
"""

from typing import Tuple, Optional, Any, Literal, Union, Callable
from functools import partial

import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np

from generals.core import game_jax
from generals.core.grid_jax import generate_grid
from generals.core.observation_jax import ObservationJax
from generals.core.game_jax import GameInfo, GameState
from generals.envs.jax_rendering_adapter import JaxGameAdapter
from generals.rewards import get_reward_fn


class VectorizedJaxEnv:
    """
    Vectorized JAX environment for Generals.io.
    
    Optimized for fast, GPU-accelerated RL training with:
    - Pure JAX grid generation (no NumPy/string conversions)
    - Vectorized reset and auto-reset
    - JIT-compiled operations
    - Different grids per environment
    
    Example:
        >>> # Generals.io mode (matches online game)
        >>> env = VectorizedJaxEnv(num_envs=128, mode='generalsio')
        >>> obs, info = env.reset(seed=42)
        
        >>> # Fixed grid mode (custom training)
        >>> env = VectorizedJaxEnv(num_envs=128, mode='fixed', grid_dims=(15, 15))
        >>> obs, info = env.reset(seed=42)
    """
    
    metadata = {
        'render_modes': ['human', 'rgb_array'],
        'render_fps': 10,
    }
    
    def __init__(
        self,
        num_envs: int,
        mode: Literal['fixed', 'generalsio'] = 'generalsio',
        grid_dims: Tuple[int, int] = (20, 20),
        pad_to: int = 24,
        mountain_density: float = 0.2,
        num_castles: Tuple[int, int] = (9, 15),
        render_mode: Optional[str] = None,
        render_env_index: int = 0,
        speed_multiplier: float = 1.0,
        reward_fn: Union[str, Callable] = 'land_difference',
        truncation: Optional[int] = None,
    ):
        """
        Args:
            num_envs: Number of parallel environments
            mode: 'generalsio' for random size like online game, 'fixed' for constant size
            grid_dims: Grid dimensions (only used in 'fixed' mode)
            pad_to: Pad grids to this size for batching
            mountain_density: Fraction of tiles that are mountains
            num_castles: (min, max) number of castles
            render_mode: 'human', 'rgb_array', or None
            render_env_index: Which env to render (0 to num_envs-1)
            speed_multiplier: Rendering speed
            reward_fn: Reward function name or callable. Options: 'land_difference', 
                'army_and_land', 'win_lose', 'land_with_win_bonus', or custom function
            truncation: Max timesteps before truncation (None = no truncation)
        """
        self.num_envs = num_envs
        self.mode = mode
        self.grid_dims = grid_dims
        self.pad_to = pad_to
        self.grid_size = (pad_to, pad_to)
        self.render_mode = render_mode
        self.render_env_index = min(render_env_index, num_envs - 1)
        self.speed_multiplier = speed_multiplier
        self.truncation = truncation
        
        # Reward function
        self.reward_fn = get_reward_fn(reward_fn)
        self._compute_rewards = jax.jit(jax.vmap(self.reward_fn))
        
        # Agents
        self.agent_names = ['Red', 'Blue']
        self.agent_colors = [(255, 107, 108), (0, 130, 255)]
        self.agent_data = {
            name: {"color": color}
            for name, color in zip(self.agent_names, self.agent_colors)
        }
        
        # Create partially applied grid generator
        self._generate_grid_fn = partial(
            generate_grid,
            mode=mode,
            grid_dims=grid_dims,
            pad_to=pad_to,
            mountain_density=mountain_density,
            num_castles_range=num_castles,
        )
        
        # JIT-compiled functions
        @jax.jit
        def generate_and_init(key):
            numeric_grid, valid = self._generate_grid_fn(key)
            return game_jax.create_initial_state(numeric_grid)
        
        self._generate_grid = generate_and_init
        self._generate_grids_batched = jax.jit(jax.vmap(generate_and_init))
        self._jitted_step = jax.jit(game_jax.batch_step)
        
        # State
        self.states = None
        self._rng_key = jrandom.PRNGKey(0)
        self.gui = None
        
        # Gym spaces (placeholders)
        self.action_space = None
        self.observation_space = None
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[ObservationJax, GameInfo]:
        """Reset all environments with different grids."""
        if seed is not None:
            self._rng_key = jrandom.PRNGKey(seed)
        
        # Generate different grid for each environment
        self._rng_key, *grid_keys = jrandom.split(self._rng_key, self.num_envs + 1)
        self.states = self._generate_grids_batched(jnp.stack(grid_keys))
        
        # Observations
        obs = self._get_observations()
        infos = jax.vmap(game_jax.get_info)(self.states)
        
        return obs, infos
    
    def step(
        self,
        actions: jnp.ndarray,
    ) -> Tuple[ObservationJax, jnp.ndarray, jnp.ndarray, jnp.ndarray, GameInfo]:
        """Step all environments, auto-reset terminated/truncated ones."""
        # Execute actions
        new_states, infos = self._jitted_step(self.states, actions)
        
        # Terminated: game ended (someone won)
        terminated = infos.is_done
        
        # Truncated: max timesteps reached (only if not terminated)
        if self.truncation is not None:
            truncated = (infos.time >= self.truncation) & ~terminated
        else:
            truncated = jnp.zeros(self.num_envs, dtype=jnp.bool_)
        
        # Reset if either terminated or truncated
        any_done = jnp.any(terminated | truncated)
        
        # Only do expensive reset if needed (but keep it pure JAX)
        if any_done:
            # Generate new grids for all environments
            self._rng_key, *reset_keys = jrandom.split(self._rng_key, self.num_envs + 1)
            fresh_grids = self._generate_grids_batched(jnp.stack(reset_keys))
            
            # Conditionally use fresh grid or continued state based on done
            done_mask = terminated | truncated
            def select_state(fresh, continued):
                ndim = fresh.ndim
                expanded_done = done_mask.reshape((-1,) + (1,) * (ndim - 1))
                return jnp.where(expanded_done, fresh, continued)
            
            self.states = jax.tree.map(select_state, fresh_grids, new_states)
        else:
            self.states = new_states
        
        # Observations and rewards
        obs = self._get_observations()
        rewards = self._compute_rewards(infos, self.states)
        
        return obs, rewards, terminated, truncated, infos
    
    def render(self):
        """Render the environment (if render_mode is set)."""
        if self.render_mode is None:
            return None
        
        if self.gui is None:
            from generals.gui import GUI
            from generals.gui.properties import GuiMode
            
            state = jax.tree.map(lambda x: x[self.render_env_index], self.states)
            info = game_jax.get_info(state)
            
            adapted_game = JaxGameAdapter(state, self.agent_names, info)
            self.gui = GUI(adapted_game, self.agent_data, GuiMode.TRAIN, self.speed_multiplier)
        
        # Update GUI with current state
        state = jax.tree.map(lambda x: x[self.render_env_index], self.states)
        info = game_jax.get_info(state)
        self.gui.properties.game.update_from_state(state, info)
        
        if self.render_mode == 'human':
            self.gui._GUI__renderer.render(fps=self.metadata['render_fps'])
            return None
        elif self.render_mode == 'rgb_array':
            return self.gui._GUI__renderer.get_rgb_array()
    
    def close(self):
        """Clean up resources."""
        if self.gui is not None:
            self.gui.close()
            self.gui = None
    
    def _get_observations(self) -> ObservationJax:
        """Get observations for all environments and both players."""
        if not hasattr(self, '_jitted_get_obs'):
            @jax.jit
            def get_both_obs(state):
                obs_p0 = game_jax.get_observation(state, 0)
                obs_p1 = game_jax.get_observation(state, 1)
                return jax.tree.map(
                    lambda p0, p1: jnp.stack([p0, p1]),
                    obs_p0,
                    obs_p1
                )
            
            self._jitted_get_obs = jax.jit(jax.vmap(get_both_obs))
        
        return self._jitted_get_obs(self.states)
