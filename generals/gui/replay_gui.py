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
        `.frame_change` (arrows or H/L — one step per key press),
        `.pause_toggle` (SPACE), `.restart` (R). For held-key auto-advance,
        use `play`, which drives the whole interactive loop.
        """
        return self._gui.tick(fps=fps or self.fps)

    def play(self, states, infos):
        """Run the interactive replay loop until the user quits (REPLAY mode).

        Tap Left/Right (or H/L) to step one frame; hold to run through frames
        at a steady rate; SPACE to play/pause; R to restart; Q to quit.

        We poll the held key on a timer (rather than relying on OS key-repeat,
        which isn't emitted reliably everywhere) so holding always advances.
        """
        n = len(states)
        frame = 0
        self.update(states[0], infos[0])

        loop_fps = 30                                       # responsive render/input rate
        play_ms = max(33, round(1000 / max(1, self.fps)))   # auto-play interval (SPACE)
        initial_delay = 250                                 # ms held before auto-advance starts
        repeat_ms = 35                                      # ms between steps while held

        held_since = None
        last_repeat = 0
        last_play = 0

        def goto(f):
            nonlocal frame
            f = max(0, min(n - 1, f))
            if f != frame:
                frame = f
                self.update(states[frame], infos[frame])

        while True:
            command = self.tick(fps=loop_fps)
            if command.quit:
                break
            now = pygame.time.get_ticks()

            if command.restart:
                goto(0)
            if command.frame_change:           # discrete taps (one per key press)
                goto(frame + command.frame_change)

            # Hold Left/Right (or H/L) to run through frames: after a short
            # initial delay, advance one frame every `repeat_ms`.
            pressed = pygame.key.get_pressed()
            direction = (
                1 if (pressed[pygame.K_RIGHT] or pressed[pygame.K_l])
                else -1 if (pressed[pygame.K_LEFT] or pressed[pygame.K_h])
                else 0
            )
            if direction:
                if held_since is None:
                    held_since = now
                elif now - held_since >= initial_delay and now - last_repeat >= repeat_ms:
                    goto(frame + direction)
                    last_repeat = now
            else:
                held_since = None

            if not self.paused and now - last_play >= play_ms:
                goto(frame + 1)
                last_play = now

        self.close()

    def close(self):
        """Close the GUI window."""
        self._gui.close()
