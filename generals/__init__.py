from gymnasium.envs.registration import register

from generals.agents.agent import Agent
from generals.core.environment import Action
from generals.core.grid import Grid, GridFactory
from generals.core.observation import Observation
from generals.envs.pettingzoo_generals import PettingZooGenerals

__all__ = [
    "Action",
    "Agent",
    "GridFactory",
    "PettingZooGenerals",
    "Grid",
    "Replay",
    "Observation",
    "GeneralsIOClientError",
]


def _register_gym_generals_envs():
    register(
        id="gym-generals-v0",
        entry_point="generals.envs.gymnasium_generals:GymnasiumGenerals",
    )

    register(
        id="gym-generals-image-v0",
        entry_point="generals.envs.initializers:gym_image_observations",
    )

    register(
        id="gym-generals-rllib-v0",
        entry_point="generals.envs.initializers:gym_rllib",
    )


_register_gym_generals_envs()
