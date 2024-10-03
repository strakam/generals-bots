from typing import Literal
from importlib.resources import files
from enum import Enum, IntEnum, StrEnum

#################
# Game Literals #
#################
PASSABLE: Literal["."] = "."
MOUNTAIN: Literal["#"] = "#"
CITY: Literal[0, 1, 2, 3, 4, 5, 6, 7, 8, 9] = 0  # CITY can be any digit 0-9


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


class Path(StrEnum):
    GENERAL_PATH = str(files("generals.assets.images") / "crownie.png")
    CITY_PATH = str(files("generals.assets.images") / "citie.png")
    MOUNTAIN_PATH = str(files("generals.assets.images") / "mountainie.png")

    # Font options are Quicksand-SemiBold.ttf, Quicksand-Medium.ttf, Quicksand-Light.ttf
    FONT_PATH = str(files("generals.assets.fonts") / "Quicksand-Medium.ttf")
