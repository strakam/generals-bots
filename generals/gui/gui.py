from typing import Any

from ..game import Game
from .event_handler import EventHandler
from .rendering import Renderer


class GUI:
    def __init__(
            self, game: Game, agent_data: dict[str, dict[str, Any]], from_replay=False
    ):
        self.renderer = Renderer(game, agent_data)
        self.event_handler = EventHandler(self.renderer, from_replay)

    def tick(self, fps=None):
        control_events = self.event_handler.handle_events()
        self.renderer.render(fps)
        return control_events
