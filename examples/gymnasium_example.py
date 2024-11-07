import gymnasium as gym

from generals.agents import RandomAgent, ExpanderAgent

# Initialize agents
agent = RandomAgent()
npc = ExpanderAgent()

# Create environment
env = gym.make("gym-generals-v0", agent=agent, npc=npc, render_mode="human")
