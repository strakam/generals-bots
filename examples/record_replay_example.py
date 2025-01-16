from generals import GridFactory, PettingZooGenerals
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

agents ={
    npc.id: npc,
    agent.id: agent
}

env = PettingZooGenerals(
    agents=[npc.id, agent.id],
    grid_factory=grid_factory,
    render_mode=None
)

# Options are used only for the next game
options = {
    "replay_file": "my_replay",  # Save replay as my_replay.pkl
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
