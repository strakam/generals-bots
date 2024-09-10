import generals.utils
import generals.agents
from generals.config import GameConfig


# Create agents - their names are then called for actions
agents = {
    "Red": generals.agents.RandomAgent("Red"),
    "Blue": generals.agents.RandomAgent("Blue")
}

agent_names = list(agents.keys())

testik = GameConfig(
    grid_size=16,
    agent_names=agent_names
)

# Run from replay - user can analyze the game and try different runs
generals.utils.run(testik, agents)
