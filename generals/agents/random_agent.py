import jax.numpy as jnp
import jax.random as jrandom

from generals.core.action import compute_valid_move_mask
from generals.core.observation import Observation

from .agent import Agent


class RandomAgent(Agent):
    """Agent that selects random valid actions."""

    def __init__(self, id: str = "Random", split_prob: float = 0.25, idle_prob: float = 0.05):
        super().__init__(id)
        self.idle_prob = idle_prob
        self.split_prob = split_prob

    def act(self, observation: Observation, key: jnp.ndarray) -> jnp.ndarray:
        k1, k2, k3 = jrandom.split(key, 3)

        mask = compute_valid_move_mask(observation.armies, observation.owned_cells, observation.mountains)
        valid = jnp.argwhere(mask, size=100, fill_value=-1)
        num_valid = jnp.sum(jnp.all(valid >= 0, axis=-1))

        # Pass if no valid moves or randomly with idle_prob
        should_pass = (num_valid == 0) | (jrandom.uniform(k1) < self.idle_prob)

        # Select random valid move
        idx = jrandom.randint(k2, (), 0, jnp.maximum(num_valid, 1))
        move = valid[jnp.minimum(idx, num_valid - 1)]

        # Random split decision
        split = jrandom.uniform(k3) < self.split_prob

        return jnp.array([should_pass, move[0], move[1], move[2], split], dtype=jnp.int32)

    def reset(self):
        pass
