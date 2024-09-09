import generals.utils
from generals.env import GameConfig


testik = GameConfig(grid_size=4)
agents = [generals.utils.Player("red"), generals.utils.Player("blue")]

generals.utils.run(testik, agents)
