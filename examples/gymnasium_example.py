from generals import gym_generals
from generals.agents import RandomAgent, ExpanderAgent

# Initialize agents
agent = RandomAgent()
npc = ExpanderAgent()

# Create environment -- render modes: {None, "human"}
env = gym_generals(agent=agent, npc=npc, render_mode="human")
observation, info = env.reset()

done = False

while not done:
    action = agent.play(observation)
    observation, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    env.render(fps=6)
