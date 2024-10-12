import gymnasium as gym
from generals import AgentFactory, GridFactory

# Initialize agents -- see generals/agents/agent_factory.py for more options
agent = AgentFactory.make_agent("expander")
npc = AgentFactory.make_agent("random")

# Initialize grid factory
grid_factory = GridFactory(
    grid_dims=(10, 10),  # Grid height and width
    mountain_density=0.2,  # Expected percentage of mountains
    city_density=0.05,  # Expected percentage of cities
    general_positions=[(1, 2), (7, 8)],  # Positions of the generals
    seed=38,  # Seed to generate the same map every time
)

env = gym.make(
    "gym-generals-v0",  # Environment name
    grid_factory=grid_factory,  # Grid factory
    npc=npc,  # NPC that will play against the agent
    agent_id="Agent",  # Agent ID
    agent_color=(67, 70, 86),  # Agent color
    render_mode="human",  # "human" mode is for rendering, None is for no rendering
)

# We can draw custom maps - see symbol explanations in README
grid = """
..#...##..
..A.#..4..
.3...1....
...###....
####...9.B
...###....
.2...5....
....#..6..
..#...##..
"""

# Options are used only for the next game
options = {
    "replay_file": "my_replay",  # Save replay as my_replay.pkl
    "grid": grid,  # Use the custom map
}

observation, info = env.reset(options=options)

terminated = truncated = False
while not (terminated or truncated):
    action = agent.act(observation)
    observation, reward, terminated, truncated, info = env.step(action)
    env.render()
