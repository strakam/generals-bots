from dataclasses import dataclass
from enum import Enum
from typing import Any

import pygame
from pygame.time import Clock

from generals.core.config import Dimension
from generals.core.game import Game


# Smallest legible cell; below this the army counts are unreadable anyway.
MIN_SQUARE_SIZE = 14
# Desktop space to leave for window chrome, menu bar and dock/taskbar.
SCREEN_MARGIN_X = 40
SCREEN_MARGIN_Y = 120


class GuiMode(Enum):
    TRAIN = "train"
    GAME = "game"
    REPLAY = "replay"


@dataclass
class Properties:
    __game: Game
    __agent_data: dict[str, dict[str, Any]]
    __mode: GuiMode
    __game_speed: float = 1.0
    __clock: Clock = Clock()
    __font_size = 18
    __show_tile_types: bool = False
    # Cell size in pixels. None auto-fits the board to the desktop, which is
    # what keeps a 22x22 competition board on screen at all.
    cell_size: int | None = None

    def __post_init__(self):
        self.__grid_height: int = self.__game.grid_dims[0]
        self.__grid_width: int = self.__game.grid_dims[1]
        self.__square_size: int = self.__resolve_square_size()
        self.__display_grid_width: int = self.__square_size * self.grid_width
        self.__display_grid_height: int = self.__square_size * self.grid_height
        self.__right_panel_width: int = 4 * Dimension.GUI_CELL_WIDTH.value

        self.__paused: bool = False

        self.__agent_fov: dict[str, bool] = {name: True for name in self.agent_data.keys()}

    def __resolve_square_size(self) -> int:
        """Pixels per cell: honour an explicit cell_size, else shrink to fit.

        The default 50px cell overflows most screens once the board gets big
        (a 22x22 competition board wants 1100px of height plus the panel), so
        with no explicit size we scale down to whatever the desktop allows and
        never scale up past the default.
        """
        default = Dimension.SQUARE_SIZE.value
        if self.cell_size is not None:
            return max(MIN_SQUARE_SIZE, int(self.cell_size))

        try:
            screen_w, screen_h = pygame.display.get_desktop_sizes()[0]
        except (pygame.error, IndexError, AttributeError):
            # No display info (headless, or display not initialised yet).
            return default

        available_w = screen_w - 4 * Dimension.GUI_CELL_WIDTH.value - SCREEN_MARGIN_X
        available_h = screen_h - SCREEN_MARGIN_Y
        fitted = min(available_w // max(1, self.grid_width),
                     available_h // max(1, self.grid_height))
        return max(MIN_SQUARE_SIZE, min(default, fitted))

    @property
    def square_size(self):
        """Pixels per board cell (see __resolve_square_size)."""
        return self.__square_size

    @property
    def game(self):
        return self.__game

    @property
    def agent_data(self):
        return self.__agent_data

    @property
    def mode(self):
        return self.__mode

    @property
    def paused(self):
        return self.__paused

    @paused.setter
    def paused(self, value: bool):
        self.__paused = value

    @property
    def game_speed(self):
        return self.__game_speed

    @game_speed.setter
    def game_speed(self, value: float):
        new_speed = min(32.0, max(0.25, value))  # clip speed
        self.__game_speed = new_speed

    @property
    def clock(self):
        return self.__clock

    @property
    def agent_fov(self):
        return self.__agent_fov

    @property
    def grid_height(self):
        return self.__grid_height

    @property
    def grid_width(self):
        return self.__grid_width

    @property
    def display_grid_width(self):
        return self.__display_grid_width

    @property
    def display_grid_height(self):
        return self.__display_grid_height

    @property
    def right_panel_width(self):
        return self.__right_panel_width

    @property
    def font_size(self):
        return self.__font_size

    def update_speed(self, multiplier: float) -> None:
        """multiplier: usually 2.0 or 0.5"""
        new_speed = self.game_speed * multiplier
        self.game_speed = new_speed

    @property
    def show_tile_types(self):
        return self.__show_tile_types

    @show_tile_types.setter
    def show_tile_types(self, value: bool):
        self.__show_tile_types = value
