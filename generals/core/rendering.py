"""
Adapter to make JAX GameState compatible with the existing GUI rendering system.

This module provides classes that wrap JAX GameState and present it with an
interface compatible with the legacy Game class, allowing the existing pygame
renderer to work with JAX states.
"""

import numpy as np
import jax.numpy as jnp
from typing import Any, Dict

from generals.core.game import GameState, GameInfo


class JaxChannelsAdapter:
    """
    Adapter that makes JAX GameState look like the old Channels object.

    The renderer expects:
    - channels.armies: [H, W] array
    - channels.ownership[agent]: [H, W] boolean array
    - channels.generals: [H, W] boolean array
    - channels.cities: [H, W] boolean array
    - channels.mountains: [H, W] boolean array
    - channels.get_visibility(agent): method returning [H, W] boolean array
    """

    def __init__(self, state: GameState, agents: list[str]):
        self._state = state
        self._agents = agents

        # Convert JAX arrays to numpy for pygame
        self.armies = np.array(state.armies)
        self.generals = np.array(state.generals)
        self.cities = np.array(state.cities)
        self.mountains = np.array(state.mountains)
        self.ownership_neutral = np.array(state.ownership_neutral)

        # Create ownership dict mapping agent names to their ownership arrays
        self.ownership = {}
        for i, agent in enumerate(agents):
            self.ownership[agent] = np.array(state.ownership[i])

    def get_visibility(self, agent_id: str) -> np.ndarray:
        """
        Get visibility mask for an agent (3x3 around owned cells).

        Uses the JAX visibility function but returns numpy array.
        """
        from generals.core import game

        # Find agent index
        agent_idx = self._agents.index(agent_id)

        # Use JAX visibility function
        visibility_jax = game.get_visibility(self._state.ownership[agent_idx])

        # Convert to numpy
        return np.array(visibility_jax)


class JaxGameAdapter:
    """
    Adapter that makes JAX GameState look like the old Game object.

    The renderer expects:
    - game.agents: list of agent names
    - game.channels: Channels object
    - game.grid_dims: (H, W) tuple
    - game.general_positions: {agent: (row, col)}
    - game.time: int
    - game.get_infos(): dict of player stats
    """

    def __init__(self, state: GameState, agents: list[str], info: GameInfo = None):
        self.agents = agents
        self.channels = JaxChannelsAdapter(state, agents)

        # Grid dimensions
        self.grid_dims = state.armies.shape

        # General positions
        self.general_positions = {}
        general_positions_array = np.array(state.general_positions)
        for i, agent in enumerate(agents):
            self.general_positions[agent] = general_positions_array[i]

        # Time
        self.time = int(state.time)

        # Store info for get_infos()
        self._info = info

    def get_infos(self) -> Dict[str, Dict[str, Any]]:
        """
        Return player stats in the format expected by the renderer.

        Returns:
            {
                agent_name: {
                    'army': int,
                    'land': int,
                },
                ...
            }
        """
        if self._info is None:
            # Compute info if not provided
            from generals.core import game
            # Create GameState from self.channels
            state = GameState(
                armies=jnp.array(self.channels.armies),
                ownership=jnp.stack([jnp.array(self.channels.ownership[agent]) for agent in self.agents]),
                ownership_neutral=jnp.array(np.logical_not(
                    np.logical_or(self.channels.ownership[self.agents[0]],
                                  self.channels.ownership[self.agents[1]])
                )),
                generals=jnp.array(self.channels.generals),
                cities=jnp.array(self.channels.cities),
                mountains=jnp.array(self.channels.mountains),
                passable=jnp.array(np.logical_not(self.channels.mountains)),
                general_positions=jnp.array([self.general_positions[agent] for agent in self.agents]),
                time=jnp.int32(self.time),
                winner=jnp.int32(-1),
            )
            self._info = game.get_info(state)

        # Convert to format expected by renderer
        result = {}
        info_np = {
            'army': np.array(self._info.army),
            'land': np.array(self._info.land),
        }

        for i, agent in enumerate(self.agents):
            result[agent] = {
                'army': int(info_np['army'][i]),
                'land': int(info_np['land'][i]),
            }

        return result

    def update_from_state(self, state: GameState, info: GameInfo = None):
        """Update the adapter with a new state."""
        self.channels = JaxChannelsAdapter(state, self.agents)
        self.time = int(state.time)
        self._info = info

        # Update general positions
        general_positions_array = np.array(state.general_positions)
        for i, agent in enumerate(self.agents):
            self.general_positions[agent] = general_positions_array[i]
