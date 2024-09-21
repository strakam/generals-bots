from generals.env import gym_generals
from generals.agents import RandomAgent
from generals.config import GameConfig

# Initialize agents - their names are then called for actions
agent = RandomAgent("Red")

game_config = GameConfig(
    grid_size=4,
    mountain_density=0.2,
    city_density=0.05,
    agent_names=[agent.name]
)

# Create environment
env = gym_generals(game_config, render_mode="none") # render_mode {"none", "human"}
observation, info = env.reset()

for i in range(50):
    print(f"Testing {i}/50 env..")
    done = False
    observation, info = env.reset()
    while not done:
        action = agent.play(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
