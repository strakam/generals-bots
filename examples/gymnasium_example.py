import gymnasium as gym

from generals import AgentFactory

# Initialize opponent agent
npc = AgentFactory.make_agent("expander")

# Create environment
env = gym.make("gym-generals-v0", npc=npc, render_mode="human")

observation, info = env.reset()
terminated = truncated = False
while not (terminated or truncated):
    action = env.action_space.sample()  # Here you put an action of your agent
    observation, reward, terminated, truncated, info = env.step(action)
    env.render()
