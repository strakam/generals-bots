from abc import ABC, abstractmethod

import jax.numpy as jnp

from generals.core.observation import Observation


class Agent(ABC):
    """Base class for JAX-compatible agents."""

    def __init__(self, id: str = "NPC"):
        self.id = id

    @abstractmethod
    def act(self, observation: Observation, key: jnp.ndarray) -> jnp.ndarray:
        """
        Select an action given an observation.

        Args:
            observation: Current game observation
            key: JAX random key for stochastic decisions

        Returns:
            Action array [pass, row, col, direction, split]
        """
        raise NotImplementedError

    def reset(self):
        """Reset agent state between episodes."""
        pass

    def __str__(self):
        return self.id
