"""
Generals.io environment for reinforcement learning.

This module provides the main environment class for running Generals.io games
with JAX. It supports vectorized execution for running many games in parallel.

Example:
    >>> import jax.random as jrandom
    >>> from generals import GeneralsEnv, get_observation
    >>> 
    >>> env = GeneralsEnv(grid_dims=(10, 10), truncation=500)
    >>> key = jrandom.PRNGKey(42)
    >>> state = env.reset(key)
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp

from generals.core import game
from generals.core.game import GameInfo, GameState, create_initial_state
from generals.core.game import step as game_step
from generals.core.grid import generate_grid
from generals.core.observation import Observation


class TimeStep(NamedTuple):
    """
    Result of a single environment step.

    Attributes:
        observation: Observations for both players, stacked along first axis.
        reward: Array of shape (2,) with rewards for each player.
        terminated: Boolean scalar, True if game ended (general captured).
        truncated: Boolean scalar, True if max timesteps reached.
        info: GameInfo with statistics (army counts, land counts, winner).
    """
    observation: Observation
    reward: jnp.ndarray
    terminated: jnp.ndarray
    truncated: jnp.ndarray
    info: GameInfo


class GeneralsEnv:
    """
    JAX-based Generals.io environment.

    This environment simulates the Generals.io game for two players. It supports
    vectorized execution via JAX's vmap for running thousands of games in parallel.

    Args:
        grid_dims: Tuple of (height, width) for the game grid. Default (4, 4).
        truncation: Maximum number of timesteps before game is truncated. Default 500.
        render: Whether to enable rendering (for GUI). Default False.
        mountain_density: Fraction of cells that are mountains. Default 0.15.
        num_cities_range: (min, max) number of cities to generate. Default (0, 2).
        min_generals_distance: Minimum Manhattan distance between generals. Default 3.

    Example:
        >>> # Single environment
        >>> env = GeneralsEnv(grid_dims=(10, 10), truncation=500)
        >>> state = env.reset(jrandom.PRNGKey(0))
        >>> 
        >>> # Vectorized environments with vmap
        >>> num_envs = 1024
        >>> keys = jrandom.split(jrandom.PRNGKey(0), num_envs)
        >>> states = jax.vmap(env.reset)(keys)
    """
    def __init__(
        self,
        grid_dims: tuple[int, int] = (4, 4),
        truncation: int = 500,
        render: bool = False,
        mountain_density: float = 0.15,
        num_cities_range: tuple[int, int] = (0, 2),
        min_generals_distance: int = 3,
    ):
        self.grid_dims = grid_dims
        self.truncation = truncation
        self.render = render
        self.mountain_density = mountain_density
        self.num_cities_range = num_cities_range
        self.min_generals_distance = min_generals_distance
    
    def reset(self, key: jnp.ndarray) -> GameState:
        """Reset using grid generation with mountains and cities."""
        # Explicitly set pad_to to grid size to avoid automatic padding
        h, w = self.grid_dims
        grid = generate_grid(
            key,
            grid_dims=self.grid_dims,
            pad_to=max(h, w),  # No extra padding, keep original size
            mountain_density=self.mountain_density,
            num_cities_range=self.num_cities_range,
            min_generals_distance=self.min_generals_distance,
            max_generals_distance=None,
            castle_val_range=(40, 51),
        )
        return create_initial_state(grid.astype(jnp.int32))
    
    def step(
        self,
        state: GameState,
        actions: jnp.ndarray,
        key: jnp.ndarray,
    ) -> tuple[TimeStep, GameState]:
        """
        Execute one game step.

        Args:
            state: Current game state.
            actions: Array of shape (2, 5) with actions for both players.
                Each action is [pass, row, col, direction, split].
            key: JAX random key for auto-reset on episode end.

        Returns:
            Tuple of (TimeStep, new_state). The TimeStep contains observations,
            rewards, and done flags. If the game ends, state is auto-reset.
        """
        # Step game
        new_state, info = game_step(state, actions)
        
        # Compute win/lose reward: +1 for capturing a general, -1 for losing, 0 otherwise
        reward_p0 = jnp.where(info.winner == 0, 1.0, jnp.where(info.winner == 1, -1.0, 0.0))
        rewards = jnp.array([reward_p0, -reward_p0])
        
        # Terminated: game ended (someone won)
        terminated = info.is_done
        truncated = (new_state.time >= self.truncation) & ~terminated
        
        # Auto-reset if done or truncated
        should_reset = terminated | truncated
        reset_state = self.reset(key)
        final_state = jax.tree.map(
            lambda reset, current: jnp.where(should_reset, reset, current),
            reset_state,
            new_state
        )
        
        # Get new observations
        obs_p0 = game.get_observation(final_state, 0)
        obs_p1 = game.get_observation(final_state, 1)

        # Stack observations
        observation = jax.tree.map(
            lambda p0, p1: jnp.stack([p0, p1], axis=0),
            obs_p0, obs_p1
        )
        
        timestep = TimeStep(
            observation=observation,
            reward=rewards,
            terminated=terminated,
            truncated=truncated,
            info=info
        )
        
        return timestep, final_state
