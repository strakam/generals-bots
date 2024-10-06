from .gymnasium_integration import Gym_Generals, RewardFn
from .pettingzoo_integration import PZ_Generals
from generals.agents import Agent, AgentFactory

from generals import GridFactory

"""
Here we can define environment initializers that can used
to create some "standard" environments. These environments
must be registered in the generals/__init__.py to be able
to be used in the gymnasium environment registry.
----------------------------------------------------------
Feel free to add more initializers here, maybe with some
static maps that are interesting to play on.
"""


def pz_generals_v0(
    grid_factory: GridFactory = GridFactory(),
    agents: dict[str, Agent] = None,
    reward_fn: RewardFn = None,
    render_mode=None,
):
    env = PZ_Generals(
        grid_factory=grid_factory,
        agents=agents,
        reward_fn=reward_fn,
        render_mode=render_mode,
    )
    return env


def gym_generals_v0(
    grid_factory: GridFactory = None,
    agent: Agent = None,
    npc: str = None,
    reward_fn: RewardFn = None,
    render_mode=None,
):
    env = Gym_Generals(
        grid_factory=grid_factory,
        agent=agent,
        npc=npc,
        reward_fn=reward_fn,
        render_mode=render_mode,
    )
    return env


def gym_s_v0(
    agent: Agent = None,
    npc: str = "random",
    render_mode: str = None,
    seed: float = None,
):
    grid_factory = GridFactory(
        grid_dims=(4, 4),
        mountain_density=0.2,
        city_density=0.1,
        seed=seed,
    )
    if agent is None:
        agent = AgentFactory.init_agent("empty")
    npc = AgentFactory.init_agent(npc)

    return gym_generals_v0(
        grid_factory=grid_factory,
        agent=agent,
        npc=npc,
        render_mode=render_mode,
    )


def gym_m_v0(
    agent: Agent = None,
    npc: str = "random",
    render_mode: str = None,
    seed: float = None,
):
    grid_factory = GridFactory(
        grid_dims=(8, 8),
        mountain_density=0.2,
        city_density=0.1,
        seed=seed,
    )
    agent = AgentFactory.init_agent("empty")
    npc = AgentFactory.init_agent(npc)

    return gym_generals_v0(
        grid_factory=grid_factory,
        agent=agent,
        npc=npc,
        render_mode=render_mode,
    )


def gym_l_v0(
    agent: Agent = None,
    npc: str = "random",
    render_mode: str = None,
    seed: float = None,
):
    grid_factory = GridFactory(
        grid_dims=(16, 16),
        mountain_density=0.2,
        city_density=0.1,
        seed=seed,
    )
    agent = AgentFactory.init_agent("empty")
    npc = AgentFactory.init_agent(npc)

    return gym_generals_v0(
        grid_factory=grid_factory,
        agent=agent,
        npc=npc,
        render_mode=render_mode,
    )
