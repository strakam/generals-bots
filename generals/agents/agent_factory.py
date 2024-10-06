from .random_agent import RandomAgent
from .expander_agent import ExpanderAgent
from .agent import Agent, EmptyAgent

class AgentFactory:
    """
    Factory class for creating agents.
    """

    def __init__(self):
        pass

    @staticmethod
    def init_agent(agent_type: str, **kwargs) -> Agent:
        """
        Creates an agent of the specified type.
        """
        if agent_type == "random":
            return RandomAgent(**kwargs)
        elif agent_type == "expander":
            return ExpanderAgent(**kwargs)
        elif agent_type == "empty":
            return EmptyAgent(**kwargs)
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
