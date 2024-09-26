from generals.env import pz_generals
from generals.agent import RandomAgent, ExpanderAgent
from generals.map import Mapper

# Initialize agents - their names are then called for actions
agents = [RandomAgent(), ExpanderAgent()]

mapper = Mapper(
    grid_size=4,
    mountain_density=0.2,
    city_density=0.05,
    general_positions=[(0, 0), (3, 3)],
)

map = """
A..#
.#3#
...#
##B#
"""

mapper.map = map


# Create environment
env = pz_generals(mapper, agents, render_mode="none") # render_mode {"none", "human"}

agents = {agent.name: agent for agent in agents}
observations, info = env.reset(options={"replay_file": "replay"})
done = False

while not done:
    actions = {}
    for agent in env.agents:
        # Ask agent for action
        actions[agent] = agents[agent].play(observations[agent])
    # All agents perform their actions
    observations, rewards, terminated, truncated, info = env.step(actions)
    done = any(terminated.values())
