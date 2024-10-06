import gymnasium as gym
from generals import AgentFactory

# Initialize agents
agent = AgentFactory.make_agent("expander")
npc = AgentFactory.make_agent("random")

env = gym.make(
    "gym-generals-v0",
    agent=agent,
    npc=npc,
    render_mode="human",
)

observation, info = env.reset()

terminated = truncated = False
while not (terminated or truncated):
    action = agent.act(observation)
    observation, reward, terminated, truncated, info = env.step(action)
    env.render()
