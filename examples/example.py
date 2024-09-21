from generals.env import pz_generals_v0
from generals.agents import RandomAgent
from generals.config import GameConfig

# Initialize agents - their names are then called for actions
agents = {
    "Red": RandomAgent("Red"),
    "Blue": RandomAgent("Blue")
}

game_config = GameConfig(
    grid_size=16,
    mountain_density=0.2,
    city_density=0.05,
    general_positions=[(2, 12), (8, 9)],
    agent_names=list(agents.keys()),
)

# Create environment
env = pz_generals_v0(game_config, render_mode="human") # render_mode {"none", "human"}
observations, info = env.reset(options={"replay_file": "test"})

# How fast we want rendering to be
actions_per_second = 2

while not env.game.is_done():
    actions = {}
    for agent in env.agents:
        # Ask agent for action
        actions[agent] = agents[agent].play(observations[agent])
    # All agents perform their actions
    observations, rewards, terminated, truncated, info = env.step(actions)
    env.render(tick_rate=actions_per_second)
