from generals import GridFactory, PettingZooGenerals
from generals.agents import RandomAgent, ExpanderAgent

agent = ExpanderAgent()
npc = RandomAgent()

# Initialize grid factory
grid_factory = GridFactory(
    mode="uniform",  # alternatively "generalsio", which will override other parameters
    min_grid_dims=(15, 15),  # Grid height and width are randomly selected
    max_grid_dims=(23, 23),
    mountain_density=0.2,  # Expected percentage of mountains
    city_density=0.05,  # Expected percentage of cities
    seed=38,  # Seed to generate the same map every time
)

agents = {
    agent.id: agent,
    npc.id: npc,
}

env = PettingZooGenerals(agents=[agent.id, npc.id], grid_factory=grid_factory)

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
    "replay_file": "my_replay",  # If specified, save replay as my_replay.pkl
    # "grid": grid,  # Use the custom map
}

observations, info = env.reset(options=options)
terminated = truncated = False
while not (terminated or truncated):
    actions = {}
    for agent in env.agents:
        # Ask agent for action
        actions[agent] = agents[agent].act(observations[agent])
    # All agents perform their actions
    observations, rewards, terminated, truncated, info = env.step(actions)
    env.render()
