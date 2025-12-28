from enum import Enum, IntEnum, StrEnum
from importlib.resources import files
from typing import Literal

# Game Literals
MOUNTAIN: Literal["#"] = "#"


class Dimension(IntEnum):
    SQUARE_SIZE = 50
    GUI_CELL_HEIGHT = 30
    GUI_CELL_WIDTH = 70
    MINIMUM_WINDOW_SIZE = 700


class Direction(Enum):
    UP = (-1, 0)
    DOWN = (1, 0)
    LEFT = (0, -1)
    RIGHT = (0, 1)


DIRECTIONS = [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]


class Path(StrEnum):
    GENERAL_PATH = str(files("generals.assets.images") / "crown.png")
    CITY_PATH = str(files("generals.assets.images") / "city.png")
    MOUNTAIN_PATH = str(files("generals.assets.images") / "mountain.png")

    # Font options are Quicksand-SemiBold.ttf, Quicksand-Medium.ttf, Quicksand-Light.ttf
    FONT_PATH = str(files("generals.assets.fonts") / "Quicksand-Medium.ttf")
