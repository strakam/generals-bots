import generals.utils
import generals.agents
from generals.config import GameConfig


testik = GameConfig(grid_size=4)
agents = {
    "red": generals.agents.RandomAgent("red"),
    "blue": generals.agents.RandomAgent("blue")
}

generals.utils.run(testik, agents)
