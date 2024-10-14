from gymnasium.envs.registration import register

from generals.agents import AgentFactory
from generals.core.grid import Grid, GridFactory
from generals.core.replay import Replay
from generals.envs.pettingzoo_generals import PettingZooGenerals

__all__ = [
    "AgentFactory",
    "GridFactory",
    "PettingZooGenerals",
    "Grid",
    "Replay",
]


def _register_gym_generals_envs():
    register(
        id="gym-generals-v0",
        entry_point="generals.envs.gymnasium_generals:GymnasiumGenerals",
    )

    register(
        id="gym-generals-normalized-v0",
        entry_point="generals.envs.initializers:gyms_generals_normalized_v0",
    )


_register_gym_generals_envs()
