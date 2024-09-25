from generals.env import pz_generals
from generals.agents import ExpanderAgent, RandomAgent
from generals.config import GameConfig

# Initialize agents
agents = [RandomAgent(), ExpanderAgent()]

game_config = GameConfig(
    grid_size=16,
    mountain_density=0.2,
    city_density=0.05,
    general_positions=[(4, 12), (12, 4)],
    agents=agents,
)

# Create environment
env = pz_generals(game_config, render_mode="human")  # render_mode {None, "human"}
observations, info = env.reset()

# How fast we want rendering to be
actions_per_second = 6
agents = {
    agent.name: agent for agent in agents
}  # Create a dictionary of agents - their names are then called for actions

while not env.game.is_done():
    actions = {}
    for agent in env.agents:
        # Ask agent for action
        actions[agent] = agents[agent].play(observations[agent])
    # All agents perform their actions
    observations, rewards, terminated, truncated, info = env.step(actions)
    env.render(tick_rate=actions_per_second)
