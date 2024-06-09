from typing import List, Tuple, Literal, Dict

from pydantic import BaseModel

class Config(BaseModel):

    # Game settings
    n_players: int = 2
    grid_size: int = 10
    town_density: float = 0.1
    mountain_density: float = 0.1
    starting_positions: List[Tuple[int, int]] = [(0, 0), (10, 10)]
    map_name: str = None
    tick_rate: int = 10

    # Game Literals
    PASSABLE: Literal[0] = 0
    MOUNTAIN: Literal[1] = 1
    CITY: Literal[2] = 2
    GENERAL: Literal[3] = 3
    ARMY: Literal[4] = 4
    OWNERSHIP: Literal[5] = 5

    # Moves
    UP: List[int] = [-1, 0]
    DOWN: List[int] = [1, 0]
    LEFT: List[int] = [0, -1]
    RIGHT: List[int] = [0, 1]

    # GUI constants
    SQUARE_SIZE: int = 40
    GRID_OFFSET: int = 40
    WINDOW_HEIGHT: int = SQUARE_SIZE * grid_size + GRID_OFFSET
    WINDOW_WIDTH: int = SQUARE_SIZE * grid_size

    FOG_OF_WAR: Tuple[int, int, int] = (80, 83, 86)
    VISIBLE: Tuple[int, int, int] = (200, 200, 200)
    PLAYER_1_COLOR: Tuple[int, int, int] = (255, 0, 0)
    PLAYER_2_COLOR: Tuple[int, int, int] = (0, 0, 255)
    BLACK: Tuple[int, int, int] = (0, 0, 0)
    WHITE: Tuple[int, int, int] = (255, 255, 255)
    PLAYER_COLORS: Dict[int, Tuple[int, int, int]] = {1: PLAYER_1_COLOR, 2: PLAYER_2_COLOR}




