import jax
import jax.numpy as jnp
import jax.random as jrandom
from typing import Tuple, NamedTuple, Optional

from generals.core.game import GameState, GameInfo, step as game_step, create_initial_state
from generals.core.observation import Observation
from generals.core import game


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
    def __init__(self, truncation: Optional[int] = 500):
        self.truncation = truncation
    
    def reset(self, key: jnp.ndarray) -> GameState:
        """Reset to 4x4 grid with generals in corners."""
        grid = jnp.zeros((4, 4), dtype=jnp.int32)
        grid = grid.at[0, 0].set(1).at[3, 3].set(2)
        return create_initial_state(grid)
    
    def step(
        self, 
        state: GameState, 
        actions: jnp.ndarray,
        key: jnp.ndarray
    ) -> Tuple[TimeStep, GameState]:
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
        
        # Truncated: max timesteps reached (only if not terminated)
        if self.truncation is not None:
            truncated = (new_state.time >= self.truncation) & ~terminated
        else:
            truncated = jnp.bool_(False)
        
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
