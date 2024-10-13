from gymnasium.envs.registration import register

from .agents.agent_factory import AgentFactory
from .core.grid import Grid, GridFactory
from .core.replay import Replay

__all__ = [
    "AgentFactory",
    "GridFactory",
    "Grid",
    "Replay",
]


def _register_generals_envs():
    register(
        id="gym-generals-v0",
        entry_point="generals.envs.env:gym_generals_v0",
    )

    register(
        id="pz-generals-v0",
        entry_point="generals.envs.env:pz_generals_v0",
        disable_env_checker=True,
    )


_register_generals_envs()
