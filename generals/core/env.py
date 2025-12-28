from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.random as jrandom

from generals.core import game
from generals.core.game import GameInfo, GameState, create_initial_state
from generals.core.game import step as game_step
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
    def __init__(self, truncation: int = 5000, render: bool = False):
        self.truncation = truncation
        self.render = render
    
    def reset(self, key: jnp.ndarray) -> GameState:
        """Reset to 4x4 grid with random general positions."""
        grid = jnp.zeros((4, 4), dtype=jnp.int32)
        # Sample two different random positions out of 16
        idx = jrandom.choice(key, 16, shape=(2,), replace=False)
        pos_a = (idx[0] // 4, idx[0] % 4)
        pos_b = (idx[1] // 4, idx[1] % 4)
        grid = grid.at[pos_a].set(1).at[pos_b].set(2)
        return create_initial_state(grid)
    
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
