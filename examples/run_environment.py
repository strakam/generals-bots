from generals.env import generals_v0
from generals.agents import RandomAgent
from generals.config import GameConfig

# Initialize agents - their names are then called for actions
agents = {
    "red": RandomAgent("red"),
    "blue": RandomAgent("blue")
}

agent_names = list(agents.keys())

game_config = GameConfig(
    grid_size=4,
    agent_names=agent_names
)

# Create environment
env = generals_v0(game_config, render_mode="none")
observations, info = env.reset()

while not env.game.is_done():
    actions = {}
    for agent in env.agents:
        # Ask agent for action
        actions[agent] = agents[agent].play(observations[agent])
    # All agents perform their actions
    observations, rewards, terminated, truncated, info = env.step(actions)
