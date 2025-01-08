from generals import GridFactory
from generals.agents import Agent
from generals.envs.gymnasium_generals import GymnasiumGenerals
from generals.envs.gymnasium_wrappers import ObservationAsImageWrapper, RemoveActionMaskWrapper
from generals.rewards.reward_fn import RewardFn

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


def gym_image_observations(
    grid_factory: GridFactory = None,
    npc: Agent = None,
    agent: Agent = None,
    to_render: bool = None,
    reward_fn: RewardFn = None,
):
    """
    Example of a Gymnasium environment initializer that creates
    an environment that returns image observations.
    """
    _env = GymnasiumGenerals(
        grid_factory=grid_factory,
        npc=npc,
        agent=agent,
        to_render=to_render,
        reward_fn=reward_fn,
    )
    env = ObservationAsImageWrapper(_env)
    return env


def gym_rllib(
    grid_factory: GridFactory = None,
    npc: Agent = None,
    agent: Agent = None,
    to_render: bool = None,
    reward_fn: RewardFn = None,
):
    """
    Example of a Gymnasium environment initializer that creates
    an environment that returns image observations.
    """
    env = GymnasiumGenerals(
        grid_factory=grid_factory,
        npc=npc,
        agent=agent,
        to_render=to_render,
        reward_fn=reward_fn,
    )
    image_env = ObservationAsImageWrapper(env)
    no_action_env = RemoveActionMaskWrapper(image_env)
    return no_action_env
