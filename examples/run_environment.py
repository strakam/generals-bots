from generals.env import generals_v0
from generals.agents import RandomAgent
from generals.config import GameConfig

# Initialize agents - their names are then called for actions
agents = {
    "monkas": RandomAgent("monkas"),
    "topkek": RandomAgent("topkek")
}

game_config = GameConfig(
    grid_size=4,
    mountain_density=0.2,
    city_density=0.05,
    general_positions=[(0, 0), (3, 3)],
    agent_names=list(agents.keys()),
)
# Create environment
env = generals_v0(game_config, render_mode="none")
observations, info = env.reset(options={"replay_file": "test"})
while not env.game.is_done():
    actions = {}
    for agent in env.agents:
        # Ask agent for action
        actions[agent] = agents[agent].play(observations[agent])
    # All agents perform their actions
    observations, rewards, terminated, truncated, info = env.step(actions)
