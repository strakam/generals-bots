"""Watch HunterAgent take on ExpanderAgent.

Runs one game headless (fast), then opens a pygame window in replay mode so you
can scrub it frame-by-frame.

    uv run python -m examples.hunter_vs_expander          # default seed
    uv run python -m examples.hunter_vs_expander 7        # pick a seed

Controls: SPACE play/pause | ←/→ or H/L step a frame (hold to run) | R restart | Q quit
"""
import sys

import jax.numpy as jnp
import jax.random as jrandom

from generals import GeneralsEnv, get_observation
from generals.agents import ExpanderAgent, HunterAgent
from generals.core import game as game_module
from generals.gui import ReplayGUI
from generals.gui.properties import GuiMode

GRID_DIMS = (10, 10)
TRUNCATION = 500
FPS = 8
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0

agents = [HunterAgent(id="Hunter"), ExpanderAgent(id="Expander")]
COLORS = [(50, 80, 200), (200, 50, 50)]  # Hunter blue, Expander red

# ---- Simulate headless ----
env = GeneralsEnv(grid_dims=GRID_DIMS, truncation=TRUNCATION)
pool, _ = env.reset(jrandom.PRNGKey(0))
state = env.init_state(jrandom.PRNGKey(SEED))

states_log = [state]
infos_log = [game_module.get_info(state)]
key = jrandom.PRNGKey(7)
terminated = truncated = False
while not (terminated or truncated):
    key, *subkeys = jrandom.split(key, 3)
    actions = jnp.stack([agents[i].act(get_observation(state, i), subkeys[i]) for i in range(2)])
    timestep, state = env.step(state, actions, pool)
    states_log.append(timestep.last_state)
    infos_log.append(timestep.info)
    terminated, truncated = bool(timestep.terminated), bool(timestep.truncated)

winner = int(infos_log[-1].winner)
print(f"seed {SEED}: {len(states_log) - 1} steps, "
      f"winner = {['Hunter', 'Expander', 'draw'][winner if winner >= 0 else 2]}")

# ---- Step-through replay ----
# gui.play drives the whole loop, including hold-to-run frame scrubbing
# (a manual tick() loop only sees discrete taps — OS key-repeat is off).
gui = ReplayGUI(states_log[0], agent_ids=[a.id for a in agents], colors=COLORS,
                fps=FPS, mode=GuiMode.REPLAY, start_paused=True)
print("Replay open. Controls: SPACE play/pause | ←/→ or H/L step (hold to run) | R restart | Q quit")
gui.play(states_log, infos_log)
