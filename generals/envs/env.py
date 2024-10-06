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
    assert len(agents) == 2, "Only 2 agents are supported in PZ_Generals."
    env = PZ_Generals(
        grid_factory=grid_factory,
        agents=agents,
        reward_fn=reward_fn,
        render_mode=render_mode,
    )
    return env


def gym_generals_v0(
    grid_factory: GridFactory = GridFactory(),
    agent: Agent = None,
    npc: Agent = None,
    reward_fn: RewardFn = None,
    render_mode=None,
):
    if npc is None:
        print("No npc provided, using RandomAgent.")
        npc = AgentFactory.init_agent("random")
    env = Gym_Generals(
        grid_factory=grid_factory,
        agent=agent,
        npc=npc,
        reward_fn=reward_fn,
        render_mode=render_mode,
    )
    return env
