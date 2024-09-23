from generals.env import gym_generals
import stable_baselines3.common.env_checker as env_checker
from generals.config import GameConfig

game_config = GameConfig(
    grid_size=16,
    mountain_density=0.2,
    city_density=0.05,
    general_positions=[(2, 12), (8, 9)],
    agent_names=['Red'],
)

env = gym_generals(game_config, render_mode="none")
env_checker.check_env(env)
print('SB3 check passed!')
