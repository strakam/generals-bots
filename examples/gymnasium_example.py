from generals.env import gym_generals
from generals.agents import RandomAgent
from generals.config import GameConfig

# Initialize agents - their names are then called for actions
agent = RandomAgent("Red")

game_config = GameConfig(
    grid_size=16,
    mountain_density=0.2,
    city_density=0.05,
    general_positions=[(2, 12), (8, 9)],
    agent_names=[agent.name],
    gymnasium_npc="expander"
)

# Create environment
env = gym_generals(game_config, render_mode="human") # render_mode {None, "human"}
observation, info = env.reset()

# How fast we want rendering to be
actions_per_second = 6

done = False

while not done:
    action = agent.play(observation)
    observation, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    env.render(tick_rate=actions_per_second)
