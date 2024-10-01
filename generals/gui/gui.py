from typing import Any

from ..game import Game
from .properties import Properties
from .event_handler import EventHandler
from .rendering import Renderer


class GUI:
    def __init__(
            self, game: Game, agent_data: dict[str, dict[str, Any]], from_replay=False
    ):
        self.properties = Properties()
        self.__renderer = Renderer(game, self.properties, agent_data)
        self.__event_handler = EventHandler(self.__renderer, self.properties, from_replay)

    def tick(self, fps=None):
        control_events = self.__event_handler.handle_events()
        self.__renderer.render(fps)
        return control_events
