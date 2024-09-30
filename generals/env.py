from .agents import Agent
from .integrations.gymnasium_integration import Gym_Generals, RewardFn as RewardFnGym
from .integrations.pettingzoo_integration import PZ_Generals, RewardFn as RewardFnPz

from .map import Mapper


def pz_generals(
    mapper: Mapper = Mapper(),
    agents: dict[str, Agent] = None,
    reward_fn: RewardFnPz=None,
    render_mode=None,
):
    """
    Here we apply wrappers to the environment.
    """
    env = PZ_Generals(
        mapper=mapper, agents=agents, reward_fn=reward_fn, render_mode=render_mode
    )
    return env


def gym_generals(
    mapper: Mapper = Mapper(),
    agent: Agent = None,
    npc: Agent = None,
    reward_fn: RewardFnGym=None,
    render_mode=None,
):
    """
    Here we apply wrappers to the environment.
    """
    env = Gym_Generals(
        mapper=mapper,
        agent=agent,
        npc=npc,
        reward_fn=reward_fn,
        render_mode=render_mode,
    )
    return env
