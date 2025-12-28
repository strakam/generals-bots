import jax.numpy as jnp

from generals.core.observation import Observation

from .agent import Agent
from ._expander_logic import expander_action


class ExpanderAgent(Agent):
    """Agent that aggressively expands territory by capturing new cells."""

    def __init__(self, id: str = "Expander"):
        super().__init__(id)

    def act(self, observation: Observation, key: jnp.ndarray) -> jnp.ndarray:
        return expander_action(key, observation)

    def reset(self):
        pass
