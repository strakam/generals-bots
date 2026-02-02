"""Visualize a game between two agents using the pygame GUI."""
import jax.numpy as jnp
import jax.random as jrandom

from generals import GeneralsEnv, get_observation
from generals.agents import RandomAgent, ExpanderAgent
from generals.gui import ReplayGUI

# Configuration
GRID_DIMS = (12, 12)
TRUNCATION = 500
FPS = 10

# Create environment and agents
env = GeneralsEnv(grid_dims=GRID_DIMS, truncation=TRUNCATION)
agent_0 = RandomAgent(id="Random")
agent_1 = ExpanderAgent(id="Expander")

# Initialize game
key = jrandom.PRNGKey(42)
state = env.reset(key)

# Create GUI
gui = ReplayGUI(state, agent_ids=[agent_0.id, agent_1.id])

terminated = truncated = False
step_count = 0

while not (terminated or truncated):
    obs_0 = get_observation(state, 0)
    obs_1 = get_observation(state, 1)

    key, k1, k2 = jrandom.split(key, 3)
    actions = jnp.stack([agent_0.act(obs_0, k1), agent_1.act(obs_1, k2)])

    timestep, state = env.step(state, actions)

    gui.update(state, timestep.info)
    gui.tick(fps=FPS)

    terminated = bool(timestep.terminated)
    truncated = bool(timestep.truncated)
    step_count += 1

winner = [agent_0.id, agent_1.id][int(timestep.info.winner)] if timestep.info.winner >= 0 else "None"
print(f"Game over after {step_count} steps! Winner: {winner}")

import time
time.sleep(2)
gui.close()
