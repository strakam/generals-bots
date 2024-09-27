from generals.env import pz_generals
from generals.agent import ExpanderAgent, RandomAgent

# Initialize agents
random = RandomAgent()
expander = ExpanderAgent()

agents = {
    random.name: random,
    expander.name: expander,
}  # Environment calls agents by name

# Create environment -- render modes: {None, "human"}
env = pz_generals(agents=agents, render_mode="human")
observations, info = env.reset()

while not env.game.is_done():
    actions = {}
    for agent in env.agents:
        # Ask agent for action
        actions[agent] = agents[agent].play(observations[agent])
    # All agents perform their actions
    observations, rewards, terminated, truncated, info = env.step(actions)
    env.render(fps=6)
