"""Harvester: Hunter, plus it banks the cities Hunter walks past.

A minimal improvement on the Hunter. Army income is dominated by *structures*:
the general and every owned **city** grow +1 every 2 turns, while plain land
grows only +1 per 50 turns. Hunter and Expander both route around cities and
never take them. The Harvester runs Hunter's exact garrison / conveyor / kill
machinery, but once its stack is big enough to afford a neutral castle it detours
to seize it first — every city is a permanent +0.5 army/turn engine, so it
out-masses a city-less Hunter and then decapitates it the same way.
"""
import jax
import jax.numpy as jnp

from generals.core.observation import Observation

from .agent import Agent
from .hunter_agent import GARRISON, _bfs, _toward


@jax.jit
def harvester_action(key, obs):
    """Capture general > seize an affordable city > feed surplus > advance > wait."""
    del key
    a, mine = obs.armies, obs.owned_cells
    H, W = a.shape
    reach = jnp.int32(H * W)
    mine_army = jnp.where(mine, a, 0)
    movable = mine & (a > 1)
    biggest = jnp.max(mine_army)

    gen = mine & obs.generals
    gen_army = jnp.sum(jnp.where(gen, a, 0))
    g = jnp.argmax(gen.reshape(-1).astype(jnp.int32))

    # Affordable neutral castles become walkable targets; other cities stay walls.
    affordable_city = obs.cities & obs.neutral_cells & (biggest - 1 > a)
    passable = ~(obs.mountains | obs.structures_in_fog | (obs.cities & ~mine & ~affordable_city))
    from_gen = _bfs(passable, gen)

    # Goal: enemy general > an affordable city to bank > nearest enemy land > scout.
    egen = obs.opponent_cells & obs.generals
    enemy = obs.opponent_cells & ~obs.cities
    fog = obs.fog_cells & passable & (from_gen < reach)
    open_ = passable & ~mine & (from_gen < reach)
    farthest = lambda m: m & (from_gen == jnp.max(jnp.where(m, from_gen, -1)))
    goal = jnp.where(jnp.any(egen), egen,
           jnp.where(jnp.any(affordable_city), affordable_city,
           jnp.where(jnp.any(enemy), enemy,
           jnp.where(jnp.any(fog), farthest(fog), farthest(open_)))))

    to_goal = _bfs(passable, goal)
    direction, nbr = _toward(to_goal, passable)
    advances = nbr < to_goal
    dirn = direction.reshape(-1)

    egen_army = jnp.sum(jnp.where(egen, a, 0))
    kill = jnp.any(egen) & movable & (to_goal == 1) & advances & (a - 1 > egen_army)
    ki = jnp.argmax(jnp.where(kill, mine_army, -1).reshape(-1))
    feed = (gen_army >= 2 * GARRISON) & advances.reshape(-1)[g]
    fwd = movable & ~gen & advances
    ci = jnp.argmax(jnp.where(fwd, mine_army, -1).reshape(-1))

    do_kill = jnp.any(kill)
    do_feed = ~do_kill & feed
    do_conv = ~do_kill & ~do_feed & jnp.any(fwd)
    i = jnp.where(do_kill, ki, jnp.where(do_feed, g, ci))
    return jnp.array([~(do_kill | do_feed | do_conv), i // W, i % W, dirn[i], do_feed], dtype=jnp.int32)


class HarvesterAgent(Agent):
    """Runs Hunter's playbook but banks affordable cities for a runaway economy."""

    def __init__(self, id: str = "Harvester"):
        super().__init__(id)

    def act(self, observation: Observation, key: jnp.ndarray) -> jnp.ndarray:
        return harvester_action(key, observation)

    def reset(self):
        pass
