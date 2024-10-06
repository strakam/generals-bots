from .core.grid import GridFactory, Grid
from .core.replay import Replay
from gymnasium.envs.registration import register


__all__ = [
    "GridFactory",
    "Grid",
    "Replay",
]


def _register_generals_envs():
    register(
        id="gym-s-v0",
        entry_point="generals.envs.env:gym_s_v0",
    )

    register(
        id="gym-m-v0",
        entry_point="generals.envs.env:gym_m_v0",
    )

    register(
        id="gym-l-v0",
        entry_point="generals.envs.env:gym_l_v0",
    )

    register(
        id="gym-generals-v0",
        entry_point="generals.envs.env:gym_s_v0",
    )

    register(
        id="pz-generals-v0",
        entry_point="generals.envs.env:pz_generals_v0",
    )

_register_generals_envs()
