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
    
    def __init__(
        self,
        initial_state: GameState,
        agent_ids: list[str] = None,
        colors: list[str] = None,
        fps: int = 10,
    ):
        """
        Initialize the GUI.
        
        Args:
            initial_state: Initial game state to display.
            agent_ids: Names for the two players. Default ["Player 0", "Player 1"].
            colors: Colors for the two players. Default ["red", "blue"].
            fps: Default frames per second.
        """
        self.agent_ids = agent_ids or ["Player 0", "Player 1"]
        colors = colors or ["red", "blue"]
        self.fps = fps
        
        # Create adapter and full GUI
        self._adapter = JaxGameAdapter(initial_state, self.agent_ids, get_info(initial_state))
        agent_data = {
            self.agent_ids[0]: {"color": colors[0]},
            self.agent_ids[1]: {"color": colors[1]},
        }
        self._gui = FullGUI(self._adapter, agent_data, mode=GuiMode.TRAIN)
    
    def update(self, state: GameState, info: GameInfo = None):
        """Update the display with a new game state."""
        if info is None:
            info = get_info(state)
        self._adapter.update_from_state(state, info)
    
    def tick(self, fps: int = None):
        """Render frame and handle events. Call once per game step."""
        self._gui.tick(fps=fps or self.fps)
    
    def close(self):
        """Close the GUI window."""
        self._gui.close()
