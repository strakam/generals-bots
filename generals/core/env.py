"""
Generals.io environment for reinforcement learning.

This module provides the main environment class for running Generals.io games
with JAX. It supports vectorized execution for running many games in parallel.

The environment pre-generates a pool of GameStates at reset time and uses them
for cheap auto-resets during training, avoiding expensive grid generation per step.

Example:
    >>> import jax.random as jrandom
    >>> from generals import GeneralsEnv, get_observation
    >>>
    >>> env = GeneralsEnv(grid_dims=(10, 10), truncation=500)
    >>> key = jrandom.PRNGKey(42)
    >>> state = env.reset(key)
    >>> timestep, state = env.step(state, actions)
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.random as jrandom

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

    On reset, a pool of pre-generated GameStates is created. During step, auto-reset
    indexes into this pool instead of running expensive grid generation, giving ~100x
    speedup over generating grids at runtime.

    Args:
        grid_dims: Tuple of (height, width) for the game grid. Default (4, 4).
        truncation: Maximum number of timesteps before game is truncated. Default 500.
        mountain_density: Fraction of cells that are mountains. Default 0.15.
        num_cities_range: (min, max) number of cities to generate. Default (0, 2).
        min_generals_distance: Minimum BFS (shortest path) distance between generals. Default 3.
        max_generals_distance: Maximum BFS (shortest path) distance between generals. None means
            no upper bound. Useful for curriculum learning (start close, increase over time).
        pool_size: Number of pre-generated states for auto-reset. Default 10_000.

    Example:
        >>> # Single environment
        >>> env = GeneralsEnv(grid_dims=(10, 10), truncation=500)
        >>> state = env.reset(jrandom.PRNGKey(0))
        >>> timestep, state = env.step(state, actions)
        >>>
        >>> # Vectorized environments with vmap
        >>> env = GeneralsEnv(grid_dims=(10, 10), truncation=500)
        >>> state = env.reset(jrandom.PRNGKey(0))
        >>> states = jax.vmap(env.reset)(jrandom.split(jrandom.PRNGKey(1), 1024))
    """
    def __init__(
        self,
        grid_dims: tuple[int, int] = (4, 4),
        truncation: int = 500,
        mountain_density: float = 0.15,
        num_cities_range: tuple[int, int] = (0, 2),
        min_generals_distance: int = 3,
        max_generals_distance: int | None = None,
        pool_size: int = 10_000,
    ):
        self.grid_dims = grid_dims
        self.truncation = truncation
        self.mountain_density = mountain_density
        self.num_cities_range = num_cities_range
        self.min_generals_distance = min_generals_distance
        self.max_generals_distance = max_generals_distance
        self.pool_size = pool_size
        self._pool: GameState | None = None

    def _make_single_state(self, key: jnp.ndarray) -> GameState:
        """Generate a single GameState from a random key."""
        h, w = self.grid_dims
        grid = generate_grid(
            key,
            grid_dims=self.grid_dims,
            pad_to=max(h, w),
            mountain_density=self.mountain_density,
            num_cities_range=self.num_cities_range,
            min_generals_distance=self.min_generals_distance,
            max_generals_distance=self.max_generals_distance,
            castle_val_range=(40, 51),
        )
        return create_initial_state(grid.astype(jnp.int32))

    def reset(self, key: jnp.ndarray) -> GameState:
        """
        Generate the state pool and return an initial state.

        The pool is stored internally and used for cheap auto-resets in step().
        Call this once before stepping. In vectorized settings, call this once
        (not inside vmap) to generate the shared pool, then vmap over
        init_state() to get per-env starting states.

        Args:
            key: JAX random key.

        Returns:
            A single GameState ready for gameplay, with pool_idx=0.
        """
        k_pool, k_init = jrandom.split(key)
        pool_keys = jrandom.split(k_pool, self.pool_size)
        self._pool = jax.vmap(self._make_single_state)(pool_keys)
        return self._make_single_state(k_init)

    def init_state(self, key: jnp.ndarray) -> GameState:
        """
        Generate a single initial state (without regenerating the pool).

        Useful for creating per-env starting states in vectorized settings:
            states = jax.vmap(env.init_state)(keys)

        Args:
            key: JAX random key.

        Returns:
            A single GameState with pool_idx=0.
        """
        return self._make_single_state(key)

    def step(
        self,
        state: GameState,
        actions: jnp.ndarray,
    ) -> tuple[TimeStep, GameState]:
        """
        Execute one game step with auto-reset from pre-generated pool.

        When a game ends (terminated or truncated), the state is replaced with
        the next state from the internal pool. The pool_idx in the state tracks
        which pool entry to use and is incremented on each reset.

        Args:
            state: Current game state.
            actions: Array of shape (2, 5) with actions for both players.
                Each action is [pass, row, col, direction, split].

        Returns:
            Tuple of (TimeStep, new_state). The TimeStep contains observations,
            rewards, and done flags.
        """
        assert self._pool is not None, "Call env.reset(key) before env.step() to generate the state pool."

        pool = self._pool

        # Step game
        new_state, info = game_step(state, actions)

        # Compute win/lose reward
        reward_p0 = jnp.where(info.winner == 0, 1.0, jnp.where(info.winner == 1, -1.0, 0.0))
        rewards = jnp.array([reward_p0, -reward_p0])

        # Terminated / truncated flags
        terminated = info.is_done
        truncated = (new_state.time >= self.truncation) & ~terminated
        should_reset = terminated | truncated

        # Cheap auto-reset: index into pre-generated pool
        pool_idx = new_state.pool_idx
        reset_state = jax.tree.map(lambda x: x[pool_idx % self.pool_size], pool)
        new_pool_idx = jnp.where(should_reset, pool_idx + 1, pool_idx)
        reset_state = reset_state._replace(pool_idx=new_pool_idx)
        new_state = new_state._replace(pool_idx=new_pool_idx)

        final_state = jax.tree.map(
            lambda reset, current: jnp.where(should_reset, reset, current),
            reset_state,
            new_state,
        )

        # Get observations
        obs_p0 = game.get_observation(final_state, 0)
        obs_p1 = game.get_observation(final_state, 1)
        observation = jax.tree.map(
            lambda p0, p1: jnp.stack([p0, p1], axis=0),
            obs_p0, obs_p1,
        )

        timestep = TimeStep(
            observation=observation,
            reward=rewards,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )

        return timestep, final_state
