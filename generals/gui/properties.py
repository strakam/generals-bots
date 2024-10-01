from dataclasses import dataclass

from pygame.time import Clock


@dataclass
class Properties:
    paused: bool = False
    game_speed: int = 1
    clock: Clock = Clock()
