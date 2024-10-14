from generals import GridFactory
from generals.agents import Agent
from generals.envs.gymnasium_generals import GymnasiumGenerals, RewardFn
from generals.envs.gymnasium_wrappers import NormalizedObservationWrapper

"""
Here we can define environment initialization functions that
can create interesting types of environments. In case of
Gymnasium environments, please register these functions also
in the generals/__init__.py, so they can be created via gym.make
----------------------------------------------------------
Feel free to add more initializers here. It is a good place
to create "pre-wrapped" envs, or envs with custom maps or other
custom settings.
"""


def gym_generals_normalized_v0(
    grid_factory: GridFactory | None = None,
    npc: Agent | None = None,
    agent: Agent | None = None,
    render_mode: str | None = None,
    reward_fn: RewardFn | None = None,
):
    """
    Example of a Gymnasium environment initializer that creates
    an environment that returns normalized observations.
    """
    _env = GymnasiumGenerals(
        grid_factory=grid_factory,
        npc=npc,
        agent=agent,
        render_mode=render_mode,
        reward_fn=reward_fn,
    )
    env = NormalizedObservationWrapper(_env)
    return env
