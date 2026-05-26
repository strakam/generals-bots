"""
Simple GUI wrapper for visualizing JAX game states.

This provides a simpler interface than the full GUI class,
designed for quick visualization of games.
"""
from typing import Any

import pygame

from generals.core.game import GameState, GameInfo, get_info
from generals.core.rendering import JaxGameAdapter
from .gui import GUI as FullGUI
from .properties import GuiMode


class ReplayGUI:
    """
    Simple GUI for visualizing game states.
    
    Example:
        >>> gui = ReplayGUI(state, agent_ids=["Agent1", "Agent2"])
        >>> while not done:
        >>>     state = env.step(...)
        >>>     gui.update(state)
        >>>     gui.tick(fps=10)
        >>> gui.close()
    """
    
    # Dark colors only — army counts are drawn in white on top, so all backgrounds
    # need enough contrast for white text to be readable.
    DEFAULT_COLORS = [
        (200, 50, 50),    # red
        (50, 80, 200),    # blue
        (50, 130, 60),    # green
        (130, 60, 170),   # purple
        (200, 100, 30),   # dark orange
        (140, 30, 90),    # wine
        (90, 60, 30),     # brown
        (30, 130, 130),   # teal
    ]

    def __init__(
        self,
        initial_state: GameState,
        agent_ids: list[str] = None,
        colors: list[str] = None,
        fps: int = 10,
        show_tile_types: bool = False,
        mode: GuiMode = GuiMode.TRAIN,
    ):
        """
        Initialize the GUI.

        Args:
            initial_state: Initial game state to display.
            agent_ids: Names for the players. Default ["Player 0", ..., "Player N-1"]
                inferred from initial_state.ownership.shape[0].
            colors: Colors for the players. Default cycles through a built-in palette.
            fps: Default frames per second.
            show_tile_types: If True, show tile type labels (0, -2, C40, etc.) for debugging.
        """
        n_players = int(initial_state.ownership.shape[0])
        self.agent_ids = agent_ids or [f"Player {i}" for i in range(n_players)]
        if colors is None:
            colors = [self.DEFAULT_COLORS[i % len(self.DEFAULT_COLORS)] for i in range(n_players)]
        assert len(self.agent_ids) == n_players, \
            f"agent_ids has {len(self.agent_ids)} entries but state has {n_players} players"
        assert len(colors) == n_players, \
            f"colors has {len(colors)} entries but state has {n_players} players"
        self.fps = fps

        self._adapter = JaxGameAdapter(initial_state, self.agent_ids, get_info(initial_state))
        agent_data = {aid: {"color": colors[i]} for i, aid in enumerate(self.agent_ids)}
        self._gui = FullGUI(self._adapter, agent_data, mode=mode, show_tile_types=show_tile_types)
    
    def update(self, state: GameState, info: GameInfo = None):
        """Update the display with a new game state."""
        if info is None:
            info = get_info(state)
        self._adapter.update_from_state(state, info)
    
    def tick(self, fps: int = None):
        """Render frame and handle events. Returns the Command from the event handler."""
        return self._gui.tick(fps=fps or self.fps)

    @property
    def paused(self) -> bool:
        return self._gui.properties.paused

    @paused.setter
    def paused(self, value: bool) -> None:
        self._gui.properties.paused = value
    
    def close(self):
        """Close the GUI window."""
        self._gui.close()
