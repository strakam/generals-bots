from typing import List, Tuple, Literal, Dict
import importlib.resources

#################
# Game Literals #
#################
PASSABLE: Literal[0] = 0
MOUNTAIN: Literal[1] = 1
CITY: Literal[2] = 2
GENERAL: Literal[3] = 3

#########
# Moves #
#########
UP: List[int] = [-1, 0]
DOWN: List[int] = [1, 0]
LEFT: List[int] = [0, -1]
RIGHT: List[int] = [0, 1]

##################
# Game constants #
##################
INCREMENT_RATE: int = 50

########################
# Grid visual settings #
########################
SQUARE_SIZE: int = 50
UI_ROW_HEIGHT: int = 50
LINE_WIDTH: int = 1
GUI_CELL_WIDTH: int = 100

##########
# Colors #
##########
FOG_OF_WAR: Tuple[int, int, int] = (70, 73, 76)
NEUTRAL_CASTLE: Tuple[int, int, int] = (128, 128, 128) 
VISIBLE_PATH: Tuple[int, int, int] = (200, 200, 200)
VISIBLE_MOUNTAIN: Tuple[int, int, int] = (187, 187, 187)
BLACK: Tuple[int, int, int] = (0, 0, 0)
WHITE: Tuple[int, int, int] = (230, 230, 230)
PLAYER_1_COLOR: Tuple[int, int, int] = (255, 0, 0)
PLAYER_2_COLOR: Tuple[int, int, int] = (67, 99, 216)
PLAYER_COLORS: Dict[int, Tuple[int, int, int]] = {0: PLAYER_1_COLOR, 1: PLAYER_2_COLOR}


#########
# Fonts #
#########
FONT_TYPE = "Quicksand-Medium.ttf" # Font options are Quicksand-SemiBold.ttf, Quicksand-Medium.ttf, Quicksand-Light.ttf
FONT_OFFSETS = [20, 16, 12, 8, 4]  # text position for different number of digits
FONT_SIZE = 18
try: 
    with importlib.resources.path("generals.fonts", FONT_TYPE) as path:
        FONT_PATH = path
except FileNotFoundError:
    raise FileNotFoundError(f"Font file {FONT_TYPE} not found in the fonts directory")


#########
# Icons #
#########
try:
    with importlib.resources.path("generals.images", "crownie.png") as path:
        GENERAL_PATH = path
except FileNotFoundError:
    raise FileNotFoundError("Image not found")
try:
    with importlib.resources.path("generals.images", "citie.png") as path:
        CITY_PATH = path
except FileNotFoundError:
    raise FileNotFoundError("Image not found")
try:
    with importlib.resources.path("generals.images", "mountainie.png") as path:
        MOUNTAIN_PATH = path
except FileNotFoundError:
    raise FileNotFoundError("Image not found")
