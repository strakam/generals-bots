from generals.agents import RandomAgent, ExpanderAgent
from ray.rllib.algorithms.ppo import PPOConfig
import gymnasium as gym
from ray import tune
from pprint import pprint

#########################################################################################
# Currently, it seems like RLLIB uses only gymnasium 0.x, but we support gymnasium 1.0+.#
# Therefore this example may not always work.                                           #
#########################################################################################


def env_creator(env_config):
    agent = RandomAgent() # Initialize your custom agent
    npc = ExpanderAgent() # Initialize an NPC agent
    env = gym.make("gym-generals-rllib-v0", agent=agent, npc=npc) # Create the environment
    return env

tune.register_env("generals_env", env_creator)

config = (
    PPOConfig()
    .api_stack(
    enable_rl_module_and_learner=True,
    enable_env_runner_and_connector_v2=True,
    )
    .environment("generals_env") # Use the generals environment
    .env_runners(num_env_runners=1)
)

algo = config.build()

for i in range(10):
    result = algo.train()
    result.pop("config")
    pprint(result)
