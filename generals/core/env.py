"""
Generals.io environment for reinforcement learning.

This module provides the main environment class for running Generals.io games
with JAX. It supports vectorized execution for running many games in parallel.

The environment is stateless — reset() returns a pool of pre-generated states,
and step() takes the pool as an explicit argument for cheap auto-resets.

Example:
    >>> import jax.random as jrandom
    >>> from generals import GeneralsEnv, get_observation
    >>>
    >>> env = GeneralsEnv(grid_dims=(10, 10), truncation=500)
    >>> key = jrandom.PRNGKey(42)
    >>> pool, state = env.reset(key)
    >>> timestep, state = env.step(state, actions, pool)
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.random as jrandom

from generals.core import game
from generals.core.game import GameInfo, GameState, create_initial_state, fast_forward_state
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
        last_state: GameState before auto-reset (needed for bootstrap values).
    """
    observation: Observation
    reward: jnp.ndarray
    terminated: jnp.ndarray
    truncated: jnp.ndarray
    info: GameInfo
    last_state: GameState


class GeneralsEnv:
    """
    JAX-based Generals.io environment (stateless).

    This environment simulates the Generals.io game for two players. It supports
    vectorized execution via JAX's vmap for running thousands of games in parallel.

    The env is a stateless config bag — reset() returns a pool of pre-generated
    GameStates, and step() takes the pool as an explicit argument. This avoids
    JIT recompilation issues when the pool changes (e.g. curriculum, pool refresh).

    Supports two modes:
        1. Fixed size: GeneralsEnv(grid_dims=(10, 10)) — single grid size
        2. Variable sizes: GeneralsEnv(min_grid_size=8, max_grid_size=24, pad_to=24)
           — pool contains all HxW combos in [min, max], padded with mountains to pad_to

    Example:
        >>> env = GeneralsEnv(grid_dims=(10, 10), truncation=500)
        >>> pool, state = env.reset(jrandom.PRNGKey(0))
        >>> timestep, state = env.step(state, actions, pool)
    """
    def __init__(
        self,
        grid_dims: tuple[int, int] | None = None,
        truncation: int = 500,
        mountain_density: float = 0.15,
        num_cities_range: tuple[int, int] = (9, 11),
        min_generals_distance: int = 3,
        max_generals_distance: int | None = None,
        pool_size: int = 10_000,
        castle_val_range: tuple[int, int] = (40, 51),
        # Variable grid size params (alternative to grid_dims)
        min_grid_size: int | None = None,
        max_grid_size: int | None = None,
        pad_to: int | None = None,
        skip_turns: int = 0,
    ):
        # Handle backward compat: grid_dims=(h,w) → fixed size
        if grid_dims is not None:
            h, w = grid_dims
            self.min_grid_size = h  # assume square for compat
            self.max_grid_size = max(h, w)
            self.pad_to = pad_to if pad_to is not None else max(h, w)
            self._fixed_dims = grid_dims
        elif min_grid_size is not None and max_grid_size is not None:
            assert pad_to is not None and pad_to >= max_grid_size, \
                f"pad_to ({pad_to}) must be >= max_grid_size ({max_grid_size})"
            self.min_grid_size = min_grid_size
            self.max_grid_size = max_grid_size
            self.pad_to = pad_to
            self._fixed_dims = None
        else:
            # Default: 4x4 fixed
            self.min_grid_size = 4
            self.max_grid_size = 4
            self.pad_to = 4
            self._fixed_dims = (4, 4)

        self.truncation = truncation
        self.mountain_density = mountain_density
        self.num_cities_range = num_cities_range
        self.min_generals_distance = min_generals_distance
        self.max_generals_distance = max_generals_distance
        self.pool_size = pool_size
        self.castle_val_range = castle_val_range
        self.skip_turns = skip_turns

    def _make_single_state_fixed(self, key: jnp.ndarray, h: int, w: int) -> GameState:
        """Generate a single GameState for a specific (h, w) grid size."""
        grid = generate_grid(
            key,
            grid_dims=(h, w),
            pad_to=self.pad_to,
            mountain_density=self.mountain_density,
            num_cities_range=self.num_cities_range,
            min_generals_distance=self.min_generals_distance,
            max_generals_distance=self.max_generals_distance,
            castle_val_range=self.castle_val_range,
        )
        state = create_initial_state(grid.astype(jnp.int32))
        if self.skip_turns > 0:
            state = fast_forward_state(state, self.skip_turns)
        return state

    def reset(self, key: jnp.ndarray) -> tuple[GameState, GameState]:
        """
        Generate a state pool and return (pool, init_state).

        The pool is a batched GameState with shape (pool_size, ...) used for
        cheap auto-resets during step(). The init_state is a single GameState.

        Args:
            key: JAX random key.

        Returns:
            Tuple of (pool, init_state).
        """
        k_pool, k_init, k_shuffle = jrandom.split(key, 3)

        if self._fixed_dims is not None and self.min_grid_size == self.max_grid_size:
            # Fast path: single grid size
            h, w = self._fixed_dims
            pool_keys = jrandom.split(k_pool, self.pool_size)
            make_fn = lambda k: self._make_single_state_fixed(k, h, w)
            pool = jax.vmap(make_fn)(pool_keys)
        else:
            # Variable grid sizes: generate per-combo batches, concat, shuffle
            sizes = [(h, w)
                     for h in range(self.min_grid_size, self.max_grid_size + 1)
                     for w in range(self.min_grid_size, self.max_grid_size + 1)]
            num_combos = len(sizes)
            per_combo = self.pool_size // num_combos

            pool_keys = jrandom.split(k_pool, num_combos * per_combo)

            pools = []
            for i, (h, w) in enumerate(sizes):
                combo_keys = pool_keys[i * per_combo : (i + 1) * per_combo]
                make_fn = lambda k, _h=h, _w=w: self._make_single_state_fixed(k, _h, _w)
                combo_pool = jax.vmap(make_fn)(combo_keys)
                pools.append(combo_pool)

            # Concatenate all combos into one pool
            pool = jax.tree.map(lambda *xs: jnp.concatenate(xs), *pools)

            # Shuffle so different sizes are interleaved
            actual_size = num_combos * per_combo
            perm = jrandom.permutation(k_shuffle, actual_size)
            pool = jax.tree.map(lambda x: x[perm], pool)

            # Update pool_size to actual (may differ due to integer division)
            self.pool_size = actual_size

        init_state = self._make_single_state_fixed(k_init, self.max_grid_size, self.max_grid_size)
        return pool, init_state

    def init_state(self, key: jnp.ndarray) -> GameState:
        """
        Generate a single initial state (without regenerating the pool).

        Uses max_grid_size for consistency in vectorized settings.

        Args:
            key: JAX random key.

        Returns:
            A single GameState with pool_idx=0.
        """
        return self._make_single_state_fixed(key, self.max_grid_size, self.max_grid_size)

    def step(
        self,
        state: GameState,
        actions: jnp.ndarray,
        pool: GameState,
    ) -> tuple[TimeStep, GameState]:
        """
        Execute one game step with auto-reset from pool.

        When a game ends (terminated or truncated), the state is replaced with
        the next state from the pool. The pool_idx in the state tracks
        which pool entry to use and is incremented on each reset.

        Args:
            state: Current game state.
            actions: Array of shape (2, 5) with actions for both players.
                Each action is [pass, row, col, direction, split].
            pool: Batched GameState of shape (pool_size, ...) for auto-reset.

        Returns:
            Tuple of (TimeStep, new_state). The TimeStep contains observations,
            rewards, and done flags.
        """
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
            last_state=new_state,
        )

        return timestep, final_state
