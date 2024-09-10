import generals.utils
import generals.agents
from generals.config import GameConfig


# Create agents - their names are then called for actions
agents = {
    "Red": generals.agents.RandomAgent("Red"),
    "Blue": generals.agents.RandomAgent("Blue")
}

testik = GameConfig(
    grid_size=16,
    agent_names=list(agents.keys())
)

# Run from replay - user can analyze the game and try different runs
generals.utils.run(testik, agents)
