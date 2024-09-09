import generals.utils
import generals.agents
from generals.env import GameConfig


testik = GameConfig(grid_size=4)
agents = [generals.agents.RandomAgent("red"), generals.agents.RandomAgent("blue")]

generals.utils.run(testik, agents)
