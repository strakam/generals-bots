"""Hunter: garrison the general, conveyor the surplus forward, decapitate.

Winning a game means capturing the enemy general — a timeout is a draw. The
Expander spreads into 1-army piles and never defends its general, so Hunter keeps
its own general home as a growing garrison, sends only the surplus out as a
single advancing stack, and takes the enemy general the moment a pile reaches it.
"""
from functools import partial

import jax
import jax.numpy as jnp

from generals.core.observation import Observation

from .agent import Agent

GARRISON = 4  # army kept on the general; only its surplus (above 2x) is sent out


def _bfs(passable, sources):
    """Steps from `sources` to every cell over passable terrain (large = unreachable)."""
    H, W = passable.shape
    INF = jnp.int32(H * W + 5)

    def relax(_, d):
        nb = jnp.minimum(
            jnp.minimum(jnp.roll(d, 1, 0).at[0].set(INF), jnp.roll(d, -1, 0).at[-1].set(INF)),
            jnp.minimum(jnp.roll(d, 1, 1).at[:, 0].set(INF), jnp.roll(d, -1, 1).at[:, -1].set(INF)),
        )
        return jnp.where(sources, jnp.int32(0), jnp.where(passable, jnp.minimum(d, nb + 1), INF))

    return jax.lax.fori_loop(0, H * W, relax, jnp.where(sources, jnp.int32(0), INF))


def _toward(field, passable):
    """Per cell: (direction of its lowest-`field` passable neighbour, that neighbour's value).

    Directions 0,1,2,3 = UP,DOWN,LEFT,RIGHT, matching generals.core.action.DIRECTIONS.
    """
    INF = jnp.int32(field.size + 7)

    def shift(arr, fill, s, ax):
        arr = jnp.roll(arr, s, ax)
        e = 0 if s == 1 else -1
        return arr.at[e, :].set(fill) if ax == 0 else arr.at[:, e].set(fill)

    vals = jnp.stack([
        jnp.where(shift(passable, False, s, ax), shift(field, INF, s, ax), INF)
        for s, ax in ((1, 0), (-1, 0), (1, 1), (-1, 1))
    ])
    return jnp.argmin(vals, 0).astype(jnp.int32), jnp.min(vals, 0)


class HunterAgent(Agent):
    """Agent that garrisons its general and hunts down the enemy general to win."""

    def __init__(self, id: str = "Hunter"):
        super().__init__(id)

    def reset(self):
        pass

    @partial(jax.jit, static_argnums=0)
    def act(self, observation: Observation, key: jnp.ndarray) -> jnp.ndarray:
        """Pick one action: capture the general, feed the surplus out, advance, or wait."""
        del key  # deterministic
        obs = observation
        a, mine = obs.armies, obs.owned_cells
        H, W = a.shape
        reach = jnp.int32(H * W)
        passable = ~(obs.mountains | obs.structures_in_fog | (obs.castles & ~mine))
        mine_army = jnp.where(mine, a, 0)
        movable = mine & (a > 1)

        gen = mine & obs.generals
        gen_army = jnp.sum(jnp.where(gen, a, 0))
        g = jnp.argmax(gen.reshape(-1).astype(jnp.int32))
        from_gen = _bfs(passable, gen)

        # Goal: the enemy general, else nearest enemy land, else the farthest cell to scout.
        egen = obs.opponent_cells & obs.generals
        enemy = obs.opponent_cells & ~obs.castles
        fog = obs.fog_cells & passable & (from_gen < reach)
        open_ = passable & ~mine & (from_gen < reach)
        farthest = lambda m: m & (from_gen == jnp.max(jnp.where(m, from_gen, -1)))
        goal = jnp.where(jnp.any(egen), egen,
               jnp.where(jnp.any(enemy), enemy,
               jnp.where(jnp.any(fog), farthest(fog), farthest(open_))))

        to_goal = _bfs(passable, goal)
        direction, nbr = _toward(to_goal, passable)
        advances = nbr < to_goal  # moving from here steps strictly toward the goal
        dirn = direction.reshape(-1)

        # Priorities: capture general > feed surplus out (keep garrison) > advance stack > wait.
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
