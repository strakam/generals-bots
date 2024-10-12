import gymnasium as gym
from generals import AgentFactory

# Initialize agents
npc = AgentFactory.make_agent("random")

# Create environment
env = gym.make("gym-generals-v0", npc=npc, render_mode="human")

observation, info = env.reset()
terminated = truncated = False
while not (terminated or truncated):
    action = env.action_space.sample() # Here you put your agent's action
    observation, reward, terminated, truncated, info = env.step(action)
    env.render()
