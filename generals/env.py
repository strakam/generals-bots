from .agents import Agent
from .integrations.gymnasium_integration import Gym_Generals, RewardFn
from .integrations.pettingzoo_integration import PZ_Generals

from .grid import GridFactory


def pz_generals(
    grid_factory: GridFactory = GridFactory(),
    agents: dict[str, Agent] = None,
    reward_fn: RewardFn = None,
    render_mode=None,
):
    """
    Here we apply wrappers to the environment.
    """
    env = PZ_Generals(
        grid_factory=grid_factory,
        agents=agents,
        reward_fn=reward_fn,
        render_mode=render_mode,
    )
    return env


def gym_generals(
    grid_factory: GridFactory = GridFactory(),
    agent: Agent = None,
    npc: Agent = None,
    reward_fn: RewardFn = None,
    render_mode=None,
):
    """
    Here we apply wrappers to the environment.
    """
    env = Gym_Generals(
        grid_factory=grid_factory,
        agent=agent,
        npc=npc,
        reward_fn=reward_fn,
        render_mode=render_mode,
    )
    return env
