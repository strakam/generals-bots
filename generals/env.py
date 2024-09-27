from .integrations.gymnasium_integration import Gym_Generals
from .integrations.pettingzoo_integration import PZ_Generals

from .map import Mapper


def pz_generals(
    mapper: Mapper = Mapper(),
    agents: list = None,
    reward_fn=None,
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
    agent: object = None,
    npc: object = None,
    reward_fn=None,
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
