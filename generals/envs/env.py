from generals.core.game import Action, Observation, Info
from generals.envs.gymnasium_generals import GymnasiumGenerals
from generals.envs.gymnasium_wrappers import NormalizedObservationWrapper
from generals.agents import Agent
from collections.abc import Callable
from typing import TypeAlias

from .gymnasium_generals import GymnasiumGenerals
from .pettingzoo_generals import PettingZooGenerals

# Type aliases
Reward: TypeAlias = float
RewardFn: TypeAlias = Callable[[dict[str, Observation], Action, bool, Info], Reward]
AgentID: TypeAlias = str

"""
Here we can define environment initializion functions that
can create interesting types of envrionments. In case of
Gymnasium environments, please register these functions also
in the generals/__init__.py, so they can be created via gym.make
----------------------------------------------------------
Feel free to add more initializers here. It is a good place
to create "pre-wrapped" envs, or envs with custom maps or other
custom settings.
"""


def gym_generals_normalized_v0(
    grid_factory: GridFactory | None = None,
    npc: Agent | None= None,
    render_mode: str | None = None,
    reward_fn: RewardFn | None = None,
    agent_id: str = "Agent",
    agent_color: tuple[int, int, int] = (67, 70, 86),
):
    """
    Example of a Gymnasium environment initializer that creates
    an environment that returns normalized observations.
    """
    env = GymnasiumGenerals(
        grid_factory=grid_factory,
        npc=npc,
        render_mode=render_mode,
        agent_id=agent_id,
        agent_color=agent_color,
        reward_fn=reward_fn,
    )
    env = NormalizedObservationWrapper(env)
    return env
