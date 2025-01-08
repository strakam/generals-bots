from abc import ABC, abstractmethod

from generals.core.action import Action
from generals.core.observation import Observation


class Agent(ABC):
    """
    Base class for all agents.
    """

    def __init__(self, id: str = "NPC"):
        self.id = id

    @abstractmethod
    def act(self, observation: Observation) -> Action:
        """
        This method should be implemented by the child class.
        It should receive an observation and return an action.
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        """
        This method allows the agent to reset its state.
        If not needed, just pass.
        """
        raise NotImplementedError

    def __str__(self):
        return self.id
