from .gymnasium_environment import GymnasiumGenerals
from .pettingzoo_environment import PettingZooGenerals
from .wrappers.gymnasium_wrappers import (
    NormalizeObservationWrapper,
)
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
    agent_ids: list[str] = None,
    render_mode=None,
):
    assert len(agent_ids) == 2, "For now, only 2 agents are supported in PZ_Generals."
    env = PettingZooGenerals(
        grid_factory=grid_factory,
        agent_ids=agent_ids,
        render_mode=render_mode,
    )
    return env


def gym_generals_v0(
    grid_factory: GridFactory = GridFactory(),
    npc: Agent = None,
    render_mode=None,
    agent_id: str = "Agent",
    agent_color: tuple[int, int, int] = (67, 70, 86),
):
    if not isinstance(npc, Agent):
        print(
            "NPC must be an instance of Agent class, Creating random NPC as a fallback."
        )
        npc = AgentFactory.init_agent("random")
    env = GymnasiumGenerals(
        grid_factory=grid_factory,
        npc=npc,
        render_mode=render_mode,
        agent_id=agent_id,
        agent_color=agent_color,
    )
    env = NormalizeObservationWrapper(env)
    # env = RemoveActionMaskWrapper(env)
    return env
