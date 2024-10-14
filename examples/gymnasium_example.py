import gymnasium as gym

from generals import AgentFactory

# Initialize opponent agent
agent = AgentFactory.make_agent("random")
npc = AgentFactory.make_agent("expander")

# Create environment
env = gym.make("gym-generals-v0", agent=agent, npc=npc, render_mode="human")

observation, info = env.reset()
terminated = truncated = False
while not (terminated or truncated):
    action = agent.act(observation)
    observation, reward, terminated, truncated, info = env.step(action)
    env.render()
