from generals import GridFactory
from generals.envs import PettingZooGenerals
from generals.agents import RandomAgent, ExpanderAgent

agent1 = ExpanderAgent()
agent2 = RandomAgent()

agents = {
    agent1.id: agent1,
    agent2.id: agent2,
}

# Initialize grid factory
grid_factory = GridFactory(
    min_grid_dims=(10, 10),  # Grid height and width are randomly selected
    max_grid_dims=(15, 15),
    mountain_density=0.2,  # Expected percentage of mountains
    city_density=0.05,  # Expected percentage of cities
    general_positions=[(1, 2), (7, 8)],  # Positions of the generals
    seed=38,  # Seed to generate the same map every time
)

env = PettingZooGenerals(
    agent_ids=[agent1.id, agent2.id],  # Agent names
    grid_factory=grid_factory,  # Grid factory
    to_render=True,  # Render the game
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

observations, infos = env.reset(options=options)

terminated = truncated = False
while not (terminated or truncated):
    actions = {}
    for agent in env.agent_ids:
        actions[agent] = agents[agent].act(observations[agent])
    # All agents perform their actions
    observations, rewards, terminated, truncated, info = env.step(actions)
    done = terminated or truncated
    env.render()
