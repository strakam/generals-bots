"""Run a multi-agent game, save the replay, then step through it in slow motion.

By default plays a 2v2 of Expander agents. The simulation runs headless (fast),
then a pygame window opens in replay mode so you can scrub frame-by-frame.

Controls inside the replay window:
    SPACE      pause / resume autoplay
    L          next frame
    H          previous frame
    LEFT/RIGHT slower / faster autoplay
    Q          quit
"""
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np

from generals import GeneralsEnv, get_observation
from generals.agents import ExpanderAgent
from generals.core import game as game_module
from generals.gui import ReplayGUI
from generals.gui.properties import GuiMode

# ---- Configuration ----
GRID_DIMS = (15, 15)
TRUNCATION = 1000
FPS = 6
SEED = 7

NUM_PLAYERS = 4
TEAMS = jnp.array([0, 0, 1, 1], dtype=jnp.int32)  # 2v2. Use jnp.arange(N) for FFA.

agents = [ExpanderAgent(id=f"Expander{i}(T{int(TEAMS[i])})") for i in range(NUM_PLAYERS)]

PLAYER_COLORS = [
    (200, 50, 50),    # team 0 — red
    (230, 120, 60),   # team 0 — orange (warm)
    (50, 80, 200),    # team 1 — blue
    (130, 60, 170),   # team 1 — purple (cool)
]

REPLAY_PATH = Path("/tmp/generals_replay.pkl")

# ---- Simulate ----
env = GeneralsEnv(
    grid_dims=GRID_DIMS,
    truncation=TRUNCATION,
    num_players=NUM_PLAYERS,
    teams=TEAMS,
    min_generals_distance=4,
)
key = jrandom.PRNGKey(SEED)
pool, state = env.reset(key)

states_log = [state]
infos_log = [game_module.get_info(state)]

terminated = truncated = False
step = 0
while not (terminated or truncated):
    key, *subkeys = jrandom.split(key, NUM_PLAYERS + 1)
    actions = jnp.stack([
        agents[i].act(get_observation(state, i), subkeys[i])
        for i in range(NUM_PLAYERS)
    ])
    timestep, state = env.step(state, actions, pool)
    states_log.append(timestep.last_state)
    infos_log.append(timestep.info)
    terminated = bool(timestep.terminated)
    truncated = bool(timestep.truncated)
    step += 1

winner_team = int(infos_log[-1].winner)
print(f"Simulation done: {step} steps, winning team={winner_team}, eliminated={list(map(bool, states_log[-1].eliminated))}")

# Persist to disk as numpy so the file is portable / inspectable.
to_np = lambda tree: jax.tree.map(np.asarray, tree)
with open(REPLAY_PATH, "wb") as f:
    pickle.dump({
        "states": [to_np(s) for s in states_log],
        "infos": [to_np(i) for i in infos_log],
        "agent_ids": [a.id for a in agents],
        "colors": PLAYER_COLORS,
    }, f)
print(f"Saved replay to {REPLAY_PATH} ({len(states_log)} frames)")

# ---- Step-through replay ----
gui = ReplayGUI(
    states_log[0],
    agent_ids=[a.id for a in agents],
    colors=PLAYER_COLORS,
    fps=FPS,
    mode=GuiMode.REPLAY,
    start_paused=True,
)

frame = 0
gui.update(states_log[frame], infos_log[frame])
print("Replay open. Controls: SPACE play/pause | L next | H prev | ←/→ speed | Q quit")

while True:
    command = gui.tick(fps=FPS)
    if command.quit:
        break
    if command.frame_change != 0:
        frame = max(0, min(len(states_log) - 1, frame + command.frame_change))
        gui.update(states_log[frame], infos_log[frame])
    elif not gui.paused and frame < len(states_log) - 1:
        frame += 1
        gui.update(states_log[frame], infos_log[frame])

gui.close()
