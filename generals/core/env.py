from typing import NamedTuple

import jax
import jax.numpy as jnp

from generals.core import game
from generals.core.game import GameInfo, GameState, create_initial_state
from generals.core.game import step as game_step
from generals.core.grid import generate_grid
from generals.core.observation import Observation


class TimeStep(NamedTuple):
    """Gymnasium-style TimeStep."""
    observation: Observation  # [2, ...] for both players
    reward: jnp.ndarray  # [2] rewards for both players
    terminated: jnp.ndarray  # scalar bool (game ended)
    truncated: jnp.ndarray  # scalar bool (max timesteps)
    info: GameInfo

class GeneralsEnv:
    """
    Vectorized environment for Generals.io.
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
        """Step environment forward with win/lose rewards."""
        # Store prior observations
        obs_p0_prior = game.get_observation(state, 0)
        
        # Step game
        new_state, info = game_step(state, actions)
        
        # Get new observations
        obs_p0 = game.get_observation(new_state, 0)
        obs_p1 = game.get_observation(new_state, 1)
        
        # Compute win/lose reward: +1 for capturing a general, -1 for losing, 0 otherwise
        reward_p0 = (obs_p0.generals.sum() - obs_p0_prior.generals.sum()).astype(jnp.float32)
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
