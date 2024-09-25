from generals.env import gym_generals
import gymnasium.utils.env_checker as env_checker
from generals.config import GameConfig
from generals.agents import RandomAgent

agent = RandomAgent(name="A")
npc = RandomAgent(name="B")

game_config = GameConfig(
    grid_size=16,
    mountain_density=0.2,
    city_density=0.05,
    general_positions=[(2, 12), (8, 9)],
    agents=[agent, npc],
)

env = gym_generals(game_config)
env_checker.check_env(env)
print('Gymnasium check passed!')
