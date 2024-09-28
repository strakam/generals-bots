from generals.env import gym_generals
import gymnasium.utils.env_checker as env_checker
from generals.agents import RandomAgent

agent = RandomAgent(name="A")
npc = RandomAgent(name="B")

env = gym_generals(agent=agent, npc=npc)
env_checker.check_env(env)
print('Gymnasium check passed!')
