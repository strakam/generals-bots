"""Observations against a direct reference implementation.

get_observation / get_full_observation (one player) and get_observations /
get_full_observations (all players, as env.step uses them) must equal, field for
field and dtype for dtype, a plain reference: fog is the 3x3 neighbourhood of every
cell the viewer's team holds, and every map plane is masked by it. Checked along
random-action trajectories in 1v1, four-player free-for-all and 2v2.
"""
import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import pytest

from generals import GeneralsEnv
from generals.core import game
from generals.core.action import sample_valid_action
from generals.core.observation import Observation


def reference_observation(state, p, fog):
    """Straightforward NumPy version of a player's view."""
    own_all = np.asarray(state.ownership)
    teams = np.asarray(state.teams)
    armies = np.asarray(state.armies)
    N, H, W = own_all.shape
    same = teams == teams[p]
    mate = same & (np.arange(N) != p)
    own = own_all[p]
    team = own_all[same].any(axis=0)
    enemy = own_all[~same].any(axis=0)
    allied = team & ~own
    if fog:
        visible = np.zeros((H, W), bool)
        for i, j in zip(*np.nonzero(team)):
            visible[max(i - 1, 0):i + 2, max(j - 1, 0):j + 2] = True
    else:
        visible = np.ones((H, W), bool)
    structures = np.asarray(state.mountains) | np.asarray(state.castles)
    land = own_all.sum(axis=(1, 2))
    army = (armies[None] * own_all).sum(axis=(1, 2))
    return dict(
        armies=armies * visible, generals=np.asarray(state.generals) & visible,
        castles=np.asarray(state.castles) & visible, mountains=np.asarray(state.mountains) & visible,
        neutral_cells=np.asarray(state.ownership_neutral) & visible, owned_cells=own & visible,
        opponent_cells=enemy & visible, fog_cells=~visible & ~structures, structures_in_fog=~visible & structures,
        owned_land_count=land[p], owned_army_count=army[p],
        opponent_land_count=land[~same].sum(), opponent_army_count=army[~same].sum(),
        timestep=np.asarray(state.time), allied_cells=allied & visible,
        allied_land_count=land[mate].sum(), allied_army_count=army[mate].sum(),
    )


def trajectory(env, steps, seed):
    key = jrandom.PRNGKey(seed)
    pool, state = env.reset(key)
    N = env.num_players
    step = jax.jit(lambda s, a: env.step(s, a, pool))
    out = []
    for t in range(steps):
        key, k = jrandom.split(key)
        obs = game.get_observations(state)
        acts = jnp.stack([sample_valid_action(kk, jax.tree.map(lambda x: x[i], obs))
                          for i, kk in enumerate(jrandom.split(k, N))])
        _, state = step(state, acts)
        if t % 7 == 0:
            out.append(state)
    return out


SETTINGS = [dict(), dict(num_players=4), dict(teams=[0, 0, 1, 1])]


@pytest.mark.parametrize("kw", SETTINGS, ids=["1v1", "ffa4", "2v2"])
@pytest.mark.parametrize("fog", [True, False], ids=["fog", "full"])
def test_observations_match_reference(kw, fog):
    env = GeneralsEnv(grid_dims=(12, 12), truncation=400, pool_size=4, **kw)
    one = game.get_observation if fog else game.get_full_observation
    everyone = game.get_observations if fog else game.get_full_observations
    for state in trajectory(env, 180, seed=len(kw) * 10 + fog):
        stacked = everyone(state)
        for p in range(env.num_players):
            ref = reference_observation(state, p, fog)
            single = one(state, p)
            for name in Observation._fields:
                a, b = getattr(single, name), getattr(stacked, name)[p]
                assert a.dtype == b.dtype, name
                np.testing.assert_array_equal(np.asarray(a), np.asarray(b), err_msg=f"batched {name}")
                np.testing.assert_array_equal(np.asarray(a), ref[name], err_msg=f"reference {name}")


def test_visibility_is_3x3_dilation():
    key = jrandom.PRNGKey(0)
    for _ in range(20):
        key, k = jrandom.split(key)
        own = jrandom.uniform(k, (9, 11)) < 0.08
        ref = np.zeros((9, 11), bool)
        for i, j in zip(*np.nonzero(np.asarray(own))):
            ref[max(i - 1, 0):i + 2, max(j - 1, 0):j + 2] = True
        np.testing.assert_array_equal(np.asarray(game.get_visibility(own)), ref)
