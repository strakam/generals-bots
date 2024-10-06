import gymnasium as gym
import gymnasium.utils.env_checker as env_checker
from generals.agents import AgentFactory

agent = AgentFactory.make_agent("expander", name="A")
npc = AgentFactory.make_agent("random", name="B")

env = gym.make(
    "gym-generals-v0",
    agent=agent,
    npc=npc,
)
env_checker.check_env(env.unwrapped)
print('Gymnasium check passed!')
