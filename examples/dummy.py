from generals.env import pz_generals
from generals.agents import RandomAgent
from generals.config import GameConfig

# Initialize agents - their names are then called for actions
agents = {
    "Red": RandomAgent("Red"),
    "Blue": RandomAgent("Blue")
}

game_config = GameConfig(
    grid_size=4,
    mountain_density=0.2,
    city_density=0.1,
    agent_names=list(agents.keys()),
    general_positions=[(0, 0), (3, 3)]
)

map = """
A..#
.#3#
...#
##B#
"""

# Create environment
env = pz_generals(game_config, render_mode="none") # render_mode {"none", "human"}
observations, info = env.reset(map=map, options={"replay_file": "test"})
done = False

while not done:
    actions = {}
    for agent in env.agents:
        # Ask agent for action
        actions[agent] = agents[agent].play(observations[agent])
    # All agents perform their actions
    observations, rewards, terminated, truncated, info = env.step(actions)
    done = any(terminated.values())
