from gymnasium.envs.registration import register

from generals.agents.agent_factory import AgentFactory
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
        entry_point="generals.envs.env:gym_generals_v0",
    )

_register_gym_generals_envs()
