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
        show_tile_types: bool = False,
        mode: GuiMode = GuiMode.TRAIN,
        start_paused: bool = False,
    ):
        """
        Initialize the GUI.

        Args:
            initial_state: Initial game state to display.
            agent_ids: Names for the two players. Default ["Player 0", "Player 1"].
            colors: Colors for the two players. Default ["red", "blue"].
            fps: Default frames per second.
            show_tile_types: If True, show tile type labels (0, -2, C40, etc.) for debugging.
            mode: GuiMode.TRAIN for live stepping; GuiMode.REPLAY for an interactive
                scrubable replay (pause, frame stepping — see `tick`).
            start_paused: Start paused (useful in REPLAY mode).
        """
        self.agent_ids = agent_ids or ["Player 0", "Player 1"]
        # The rendering adapters key every per-player dict (ownership, generals,
        # colors, fov, stats) by name, so two identically-named agents (e.g. a
        # self-play match) would collapse both players onto one entry. Keep names
        # distinct for display so each player keeps its own slot.
        if self.agent_ids[0] == self.agent_ids[1]:
            self.agent_ids = [f"{self.agent_ids[0]} (P0)", f"{self.agent_ids[1]} (P1)"]
        colors = colors or ["red", "blue"]
        self.fps = fps

        # Create adapter and full GUI
        self._adapter = JaxGameAdapter(initial_state, self.agent_ids, get_info(initial_state))
        agent_data = {
            self.agent_ids[0]: {"color": colors[0]},
            self.agent_ids[1]: {"color": colors[1]},
        }
        self._gui = FullGUI(self._adapter, agent_data, mode=mode, show_tile_types=show_tile_types)
        self._gui.properties.paused = start_paused

    @property
    def paused(self) -> bool:
        """Whether replay playback is paused (toggled by SPACE in REPLAY mode)."""
        return self._gui.properties.paused

    def update(self, state: GameState, info: GameInfo = None):
        """Update the display with a new game state."""
        if info is None:
            info = get_info(state)
        self._adapter.update_from_state(state, info)

    def tick(self, fps: int = None):
        """Render one frame, handle input, and return the input Command.

        In REPLAY mode the command carries navigation intent: `.quit` (Q),
        `.frame_change` (H/L = back/forward one frame), `.pause_toggle` (SPACE),
        `.restart` (R), `.speed_change` (left/right arrows).
        """
        return self._gui.tick(fps=fps or self.fps)

    def close(self):
        """Close the GUI window."""
        self._gui.close()
