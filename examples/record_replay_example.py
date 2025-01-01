import gymnasium as gym

from generals import GridFactory
from generals.agents import RandomAgent, ExpanderAgent

# Initialize agents
npc = ExpanderAgent()
agent = RandomAgent()

# Initialize grid factory
grid_factory = GridFactory(
    min_grid_dims=(4, 4),  # Grid height and width
    max_grid_dims=(4, 4),
    mountain_density=0.0,  # Expected percentage of mountains
    city_density=0.05,  # Expected percentage of cities
    general_positions=[(0, 0), (3, 3)],  # Positions of the generals
    # seed=38,  # Seed to generate the same map every time
)

env = gym.make(
    "gym-generals-v0",  # Environment name
    grid_factory=grid_factory,  # Grid factory
    agent=agent,
    npc=npc,  # NPC that will play against the agent
)

# Options are used only for the next game
options = {
    "replay_file": "my_replay",  # Save replay as my_replay.pkl
}

observation, info = env.reset(options=options)

terminated = truncated = False
while not (terminated or truncated):
    action = agent.act(observation)
    observation, reward, terminated, truncated, info = env.step(action)
    env.render()
