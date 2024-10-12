import gymnasium as gym
from generals import AgentFactory, GridFactory

# Initialize agents -- see generals/agents/agent_factory.py for more options
agent = AgentFactory.make_agent("expander")
npc = AgentFactory.make_agent("random")

# Initialize grid factory
grid_factory = GridFactory(
    grid_dims=(5, 5),                   # Grid height and width
    mountain_density=0.2,               # Expected percentage of mountains
    city_density=0.05,                  # Expected percentage of cities
    general_positions=[(1, 2), (3, 4)], # Positions of the generals
    seed=38                             # Seed to generate the same map every time
)

env = gym.make(
    "gym-generals-v0",          # Environment name
    grid_factory=grid_factory,  # Grid factory
    agent_id="Agent",           # Agent ID
    agent_color=(67, 70, 86),   # Agent color
    npc=npc,                    # NPC that will play against the agent
)

# Options are used only for the next game
options = {
    "replay_file": "my_replay", # Save replay as my_replay.pkl
}

observation, info = env.reset(options=options)

terminated = truncated = False
while not (terminated or truncated):
    action = agent.act(observation)
    observation, reward, terminated, truncated, info = env.step(action)
    env.render()
