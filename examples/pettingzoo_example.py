from generals.env import pz_generals
from generals.agent import ExpanderAgent, RandomAgent
from generals.map import Mapper

# Initialize agents
agents = [RandomAgent(), ExpanderAgent()]
mapper = Mapper(
    grid_size=16,
    mountain_density=0.2,
    city_density=0.05,
    general_positions=[(4, 12), (12, 4)],
)

# Create environment
env = pz_generals(mapper, agents, render_mode="human")  # render_mode {None, "human"}
observations, info = env.reset(options={"replay_file": "replay"})

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
