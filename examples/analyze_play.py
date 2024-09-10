import generals.utils
import generals.agents
from generals.config import GameConfig


# Create agents - their names are then called for actions
agents = {
    "red": generals.agents.RandomAgent("red"),
    "blue": generals.agents.RandomAgent("blue")
}

agent_names = list(agents.keys())

testik = GameConfig(
    grid_size=4,
    replay_file="test",
    agent_names=agent_names
)

# Run from replay - user can analyze the game and try different runs
generals.utils.run(testik, agents)
