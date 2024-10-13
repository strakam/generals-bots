from pprint import pprint
from generals.envs import gym_generals_v0
from generals import GridFactory

from ray.rllib.algorithms import ppo, PPOConfig
from ray.tune.registry import register_env

def my_env_creator(env_config):
    gf = GridFactory(
        grid_dims=(4,4),
        seed=8
    )
    if "human" in env_config:
        return gym_generals_v0(grid_factory=gf, render_mode="human")
    return gym_generals_v0(grid_factory=gf)

register_env("gggg", my_env_creator)

config = (  # 1. Configure the algorithm,
    PPOConfig()
    .environment("gggg")
    .env_runners(num_env_runners=2)
    .framework("torch")
)

algo = config.build()  # 2. Build the algorithm,

for _ in range(50):
    pprint(algo.train())  # 3. train it,

env = my_env_creator({"human": True})
observation, info = env.reset()

terminated = truncated = False
while not (terminated or truncated):
    action = algo.compute_single_action(observation)
    observation, reward, terminated, truncated, info = env.step(action)
    env.render()


