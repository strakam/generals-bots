from generals.env import gym_generals
from generals.agents import RandomAgent, ExpanderAgent
from generals.config import GameConfig

# Initialize agents
agent = RandomAgent()
npc = ExpanderAgent()

agents = [agent, npc]  # First is player, second is NPC

game_config = GameConfig(
    grid_size=16,
    mountain_density=0.2,
    city_density=0.05,
    general_positions=[(2, 12), (8, 9)],
    agents=agents,
)

# Create environment
env = gym_generals(game_config, render_mode="human")  # render_mode {None, "human"}
observation, info = env.reset()

# How fast we want rendering to be
actions_per_second = 6
agents = {
    agent.name: agent for agent in game_config.agents
} # Create a dictionary of agents - their names are then called for actions

done = False

while not done:
    action = agent.play(observation)
    observation, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    env.render(tick_rate=actions_per_second)
