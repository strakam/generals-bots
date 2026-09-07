"""N-player rules: free-for-all and teams (2v2) through the same interface.

The engine is parameterized by the number of players N and a `teams` array.
Free-for-all is teams = arange(N); 2v2 is teams = [0, 0, 1, 1]. These tests pin
the rules the multiplayer modes add on top of the 1v1 game:

  - a move onto a teammate's cell pools the armies and hands the cell to the mover
  - capturing a general transfers ALL of the victim's cells to the capturer with
    every army halved (rounded up), turns the general into a castle and
    eliminates the victim; the game goes on while another team is alive
  - a team loses only when every one of its generals has fallen; the last team
    standing wins, and its players all get +1 while everyone else gets -1
  - an eliminated player's actions are ignored
  - sight is shared within a team
  - the 1v1 defaults are untouched: (2, 5) actions, (2, H, W) ownership
"""
import jax
import jax.numpy as jnp
import jax.random as jrandom
import pytest

from generals import GeneralsEnv
from generals.core import game
from generals.core.action import sample_valid_action
from generals.modifiers import build_castles as bc

PASS = jnp.array([1, 0, 0, 0, 0], dtype=jnp.int32)
UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
FFA3 = jnp.arange(3, dtype=jnp.int32)
FFA4 = jnp.arange(4, dtype=jnp.int32)
TEAMS_2V2 = jnp.array([0, 0, 1, 1], dtype=jnp.int32)


def move(i, j, d, split=0):
    return jnp.array([0, i, j, d, split], dtype=jnp.int32)


def board(generals: dict, size=6, teams=None):
    """Open board; generals = {player_idx: (row, col)}."""
    grid = jnp.zeros((size, size), dtype=jnp.int32)
    for p, (i, j) in generals.items():
        grid = grid.at[i, j].set(p + 1)
    return game.create_initial_state(grid, teams=teams, num_players=None if teams is not None else len(generals))


def give(state, player, ij, army):
    i, j = ij
    return state._replace(
        armies=state.armies.at[i, j].set(army),
        ownership=state.ownership.at[:, i, j].set(False).at[player, i, j].set(True),
        ownership_neutral=state.ownership_neutral.at[i, j].set(False),
    )


def passes(n):
    return jnp.stack([PASS] * n)


# ------------------------------------------------------------- state shapes


def test_state_shapes_scale_with_players():
    s = board({0: (0, 0), 1: (0, 7), 2: (7, 0), 3: (7, 7)}, size=8, teams=FFA4)
    assert s.ownership.shape == (4, 8, 8)
    assert s.general_positions.shape == (4, 2)
    assert s.teams.shape == (4,) and s.eliminated.shape == (4,)
    assert not bool(s.eliminated.any())
    assert int(s.generals.sum()) == 4
    info = game.get_info(s)
    assert info.army.shape == (4,) and info.land.shape == (4,)
    assert info.land.tolist() == [1, 1, 1, 1]


def test_default_is_the_two_player_game():
    grid = jnp.zeros((4, 4), dtype=jnp.int32).at[0, 0].set(1).at[3, 3].set(2)
    s = game.create_initial_state(grid)
    assert s.ownership.shape == (2, 4, 4)
    assert s.teams.tolist() == [0, 1]
    assert s.general_positions.tolist() == [[0, 0], [3, 3]]


def test_castle_values_are_read_above_the_general_range():
    grid = jnp.zeros((4, 4), dtype=jnp.int32).at[0, 0].set(1).at[0, 3].set(2).at[3, 0].set(3).at[3, 3].set(4)
    grid = grid.at[1, 1].set(40)
    s = game.create_initial_state(grid, teams=FFA4)
    assert bool(s.castles[1, 1]) and int(s.armies[1, 1]) == 40
    assert int(s.castles.sum()) == 1                 # generals 3 and 4 are not castles
    assert bool(s.ownership[2, 3, 0]) and bool(s.ownership[3, 3, 3])


def test_missing_general_starts_eliminated():
    grid = jnp.zeros((4, 4), dtype=jnp.int32).at[0, 0].set(1).at[3, 3].set(2)
    s = game.create_initial_state(grid, teams=FFA3)  # player 2 has no general on this board
    assert s.eliminated.tolist() == [False, False, True]


# ------------------------------------------------------------ friendly merge


def test_teammate_merge_pools_armies_and_transfers_ownership():
    s = board({0: (0, 0), 1: (0, 5), 2: (5, 0), 3: (5, 5)}, teams=TEAMS_2V2)
    s = s._replace(armies=s.armies.at[0, 0].set(5))
    s = give(s, 1, (0, 1), 3)                           # teammate's cell next to P0's general
    ns, _ = game.step(s, jnp.stack([move(0, 0, RIGHT), PASS, PASS, PASS]))
    assert int(ns.armies[0, 1]) == 3 + 4                 # armies pool
    assert int(ns.armies[0, 0]) == 1
    assert bool(ns.ownership[0, 0, 1]) and not bool(ns.ownership[1, 0, 1])   # cell is the mover's now
    assert not bool(ns.ownership_neutral[0, 1])
    assert int(ns.winner) == -1


def test_teammate_merge_onto_an_allied_general_is_not_a_capture():
    s = board({0: (0, 0), 1: (0, 1), 2: (5, 0), 3: (5, 5)}, teams=TEAMS_2V2)
    s = s._replace(armies=s.armies.at[0, 0].set(10))
    ns, info = game.step(s, jnp.stack([move(0, 0, RIGHT), PASS, PASS, PASS]))
    assert int(ns.armies[0, 1]) == 1 + 9                 # pooled (t=1 is an odd tick: no growth yet)
    assert bool(ns.generals[0, 1]) and not bool(ns.castles[0, 1])
    assert not bool(ns.eliminated.any())
    assert bool(ns.ownership[0, 0, 1])                   # the general tile is now held by P0, still P1's general
    assert int(info.winner) == -1


def test_enemy_cells_are_still_attacked_in_team_play():
    s = board({0: (0, 0), 1: (0, 5), 2: (5, 0), 3: (5, 5)}, teams=TEAMS_2V2)
    s = s._replace(armies=s.armies.at[0, 0].set(10))
    s = give(s, 2, (0, 1), 3)                           # enemy cell
    ns, _ = game.step(s, jnp.stack([move(0, 0, RIGHT), PASS, PASS, PASS]))
    assert int(ns.armies[0, 1]) == 9 - 3
    assert bool(ns.ownership[0, 0, 1]) and not bool(ns.ownership[2, 0, 1])


# ------------------------------------------------------------ general capture


def test_ffa_capture_transfers_everything_halved_and_eliminates():
    s = board({0: (0, 0), 1: (0, 1), 2: (4, 4)}, size=5, teams=FFA3)
    s = s._replace(armies=s.armies.at[0, 0].set(50).at[0, 1].set(3))
    s = give(s, 1, (0, 2), 20)
    s = give(s, 1, (1, 2), 5)
    s = give(s, 1, (2, 2), 1)
    s = give(s, 1, (3, 3), 0)
    ns, info = game.step(s, jnp.stack([move(0, 0, RIGHT), PASS, PASS]))

    # the general tile keeps the attacker's surplus (49 - 3), no halving there
    assert int(ns.armies[0, 1]) == 46
    # every other cell of the victim: halved, rounded up  (20->10, 5->3, 1->1, 0->0)
    assert [int(ns.armies[i]) for i in [(0, 2), (1, 2), (2, 2), (3, 3)]] == [10, 3, 1, 0]
    # ...and all of them belong to the capturer now
    for ij in [(0, 1), (0, 2), (1, 2), (2, 2), (3, 3)]:
        assert bool(ns.ownership[0][ij]) and not bool(ns.ownership[1][ij])
    assert not bool(ns.ownership[1].any())
    # the general became a castle
    assert bool(ns.castles[0, 1]) and not bool(ns.generals[0, 1])
    assert ns.eliminated.tolist() == [False, True, False]
    # P2 is still alive: no winner, play goes on and the clock advanced
    assert int(info.winner) == -1 and not bool(info.is_done)
    assert int(ns.time) == 1
    assert info.land.tolist() == [6, 0, 1]


def test_capture_needs_a_strictly_larger_force():
    s = board({0: (0, 0), 1: (0, 1), 2: (4, 4)}, size=5, teams=FFA3)
    s = s._replace(armies=s.armies.at[0, 0].set(4).at[0, 1].set(3))   # moves 3 onto 3: a tie holds
    ns, _ = game.step(s, jnp.stack([move(0, 0, RIGHT), PASS, PASS]))
    assert bool(ns.generals[0, 1]) and bool(ns.ownership[1, 0, 1])
    assert not bool(ns.eliminated.any())
    assert int(ns.armies[0, 1]) == 0


def test_captured_castle_is_still_produced_by_its_new_owner():
    s = board({0: (0, 0), 1: (0, 1), 2: (4, 4)}, size=5, teams=FFA3)
    s = s._replace(armies=s.armies.at[0, 0].set(10).at[0, 1].set(3))
    ns, _ = game.step(s, jnp.stack([move(0, 0, RIGHT), PASS, PASS]))   # time -> 1 (odd: no growth)
    assert int(ns.armies[0, 1]) == 6
    ns, _ = game.step(ns, passes(3))                                   # time -> 2 (even: +1 on structures)
    assert int(ns.armies[0, 1]) == 7


def test_eliminated_players_actions_are_ignored():
    s = board({0: (0, 0), 1: (0, 1), 2: (4, 4)}, size=5, teams=FFA3)
    s = s._replace(armies=s.armies.at[0, 0].set(50).at[0, 1].set(3))
    s = give(s, 1, (2, 2), 20)
    ns, _ = game.step(s, jnp.stack([move(0, 0, RIGHT), PASS, PASS]))
    assert bool(ns.eliminated[1])
    # P1 tries to move the stack it used to own; the cell is P0's now, so the
    # turn plays out exactly as if P1 had passed.
    quiet, _ = game.step(ns, passes(3))
    ns2, _ = game.step(ns, jnp.stack([PASS, move(2, 2, RIGHT), PASS]))
    assert jnp.array_equal(ns2.armies, quiet.armies)
    assert jnp.array_equal(ns2.ownership, quiet.ownership)
    assert bool(ns2.ownership[0, 2, 2]) and not bool(ns2.ownership[1].any())
    # Even a hand-built state where an eliminated player still "owns" a cell can't move it.
    hacked = give(ns, 1, (3, 3), 9)
    ns3, _ = game.step(hacked, jnp.stack([PASS, move(3, 3, LEFT), PASS]))
    assert int(ns3.armies[3, 2]) == 0 and int(ns3.armies[3, 3]) == 9


def test_first_capture_in_a_turn_stands():
    """Both generals are struck on the same turn: the first move to resolve
    captures and confiscates the other side's army, so the second strike never
    happens. (Under the competition ruleset the deathtouch modifier turns this
    into a draw — see tests/test_mutual_capture.py.)"""
    s = board({0: (0, 0), 1: (0, 5)}, size=6)
    s = give(s, 0, (0, 4), 40)   # strikes P1's general
    s = give(s, 1, (1, 0), 30)   # strikes P0's general; smaller army -> resolves first
    ns, info = game.step(s, jnp.stack([move(0, 4, RIGHT), move(1, 0, UP)]))
    assert int(info.winner) == 1 and bool(info.is_done)
    assert ns.eliminated.tolist() == [True, False]
    assert bool(ns.generals[0, 5])                       # P1's general was never taken


# ------------------------------------------------------------- team victory


def test_2v2_loses_only_when_both_generals_fall_and_allies_share_the_result():
    s = board({0: (0, 0), 1: (5, 5), 2: (0, 1), 3: (0, 2)}, teams=TEAMS_2V2)
    s = s._replace(armies=s.armies.at[0, 0].set(20))

    ns, info = game.step(s, jnp.stack([move(0, 0, RIGHT), PASS, PASS, PASS]))
    assert ns.eliminated.tolist() == [False, False, True, False]
    assert int(info.winner) == -1 and not bool(info.is_done)   # P3 keeps team 1 alive

    ns = ns._replace(armies=ns.armies.at[0, 1].set(20))
    ns, info = game.step(ns, jnp.stack([move(0, 1, RIGHT), PASS, PASS, PASS]))
    assert ns.eliminated.tolist() == [False, False, True, True]
    assert int(info.winner) == 0 and bool(info.is_done)        # team 0: P0 and its idle teammate P1
    assert int(ns.time) == 2
    # P1 never moved but wins with its team; the whole map is team 0's
    assert int(ns.ownership[0].sum() + ns.ownership[1].sum()) == int((~ns.ownership_neutral & ns.passable).sum())


def test_ffa_last_player_standing_wins():
    s = board({0: (0, 0), 1: (0, 1), 2: (0, 2)}, size=4, teams=FFA3)
    s = s._replace(armies=s.armies.at[0, 0].set(20))
    s, info = game.step(s, jnp.stack([move(0, 0, RIGHT), PASS, PASS]))
    assert int(info.winner) == -1
    s = s._replace(armies=s.armies.at[0, 1].set(20))
    s, info = game.step(s, jnp.stack([move(0, 1, RIGHT), PASS, PASS]))
    assert int(info.winner) == 0 and bool(info.is_done)
    assert s.eliminated.tolist() == [False, True, True]


def test_winner_is_a_team_id():
    """Team 1 wins: winner is 1 even though the capturing player is index 3."""
    s = board({0: (0, 0), 1: (0, 1), 2: (5, 5), 3: (0, 2)}, teams=TEAMS_2V2)
    s = s._replace(armies=s.armies.at[0, 2].set(20))
    s, info = game.step(s, jnp.stack([PASS, PASS, PASS, move(0, 2, LEFT)]))   # P3 takes P1
    assert int(info.winner) == -1
    s = s._replace(armies=s.armies.at[0, 1].set(20))
    s, info = game.step(s, jnp.stack([PASS, PASS, PASS, move(0, 1, LEFT)]))   # P3 takes P0
    assert int(info.winner) == 1 and bool(info.is_done)


# --------------------------------------------------------- team visibility


def test_team_shared_visibility():
    s = board({0: (0, 0), 1: (0, 5), 2: (5, 0), 3: (5, 5)}, teams=TEAMS_2V2)
    obs = game.get_observation(s, 0)
    assert not bool(obs.fog_cells[0, 4])              # next to the teammate's general: visible
    assert bool(obs.allied_cells[0, 5])               # the teammate's general shows as allied
    assert not bool(obs.owned_cells[0, 5]) and not bool(obs.opponent_cells[0, 5])
    assert bool(obs.fog_cells[3, 3])                  # the middle of the board is still fog
    assert bool(obs.fog_cells[5, 1]) and bool(obs.fog_cells[4, 5])   # enemies are not shared

    # The teammate's view is the mirror image.
    obs1 = game.get_observation(s, 1)
    assert not bool(obs1.fog_cells[0, 1]) and bool(obs1.allied_cells[0, 0])

    # Enemies see nothing of it.
    obs2 = game.get_observation(s, 2)
    assert bool(obs2.fog_cells[0, 1]) and bool(obs2.fog_cells[0, 4])
    assert not bool(obs2.allied_cells.any() & False)  # allied plane exists
    assert bool(obs2.allied_cells[5, 5])


def test_observation_counts_split_own_allied_opponent():
    s = board({0: (0, 0), 1: (0, 5), 2: (5, 0), 3: (5, 5)}, teams=TEAMS_2V2)
    s = give(s, 1, (2, 2), 7)
    s = give(s, 2, (3, 3), 4)
    s = give(s, 3, (4, 4), 6)
    obs = game.get_observation(s, 0)
    assert int(obs.owned_land_count) == 1 and int(obs.owned_army_count) == 1
    assert int(obs.allied_land_count) == 2 and int(obs.allied_army_count) == 1 + 7
    assert int(obs.opponent_land_count) == 4 and int(obs.opponent_army_count) == 1 + 4 + 1 + 6


def test_ffa_observation_matches_the_two_player_semantics():
    """No teammates: allied is empty and opponent covers everyone else."""
    s = board({0: (0, 0), 1: (0, 5), 2: (5, 0)}, teams=FFA3)
    s = give(s, 1, (0, 1), 7)
    obs = game.get_observation(s, 0)
    assert not bool(obs.allied_cells.any())
    assert int(obs.allied_land_count) == 0 and int(obs.allied_army_count) == 0
    assert bool(obs.opponent_cells[0, 1])
    assert int(obs.opponent_land_count) == 3 and int(obs.opponent_army_count) == 9
    assert obs.as_tensor().shape == (14, 6, 6)
    assert obs.as_tensor(include_allied=True).shape == (17, 6, 6)


def test_observations_have_one_shape_for_every_player():
    env = GeneralsEnv(grid_dims=(10, 10), teams=TEAMS_2V2, truncation=20, pool_size=4)
    pool, state = env.reset(jrandom.PRNGKey(0))
    ts, _ = env.step(state, passes(4), pool)
    for name, field in ts.observation._asdict().items():
        assert field.shape[0] == 4, name
    for i in range(4):
        obs_i = jax.tree.map(lambda x: x[i], ts.observation)
        assert obs_i.as_tensor(include_allied=True).shape == (17, 10, 10)


# --------------------------------------------------------------------- env


def test_env_defaults_are_unchanged():
    env = GeneralsEnv(grid_dims=(6, 6), truncation=10, pool_size=4)
    assert env.num_players == 2 and env.teams.tolist() == [0, 1]
    pool, state = env.reset(jrandom.PRNGKey(0))
    assert state.ownership.shape == (2, 6, 6)
    ts, _ = env.step(state, passes(2), pool)
    assert ts.reward.shape == (2,) and ts.observation.owned_cells.shape == (2, 6, 6)
    assert ts.observation.armies.shape == (2, 6, 6)


@pytest.mark.parametrize("kw", [dict(num_players=4), dict(teams=[0, 0, 1, 1]), dict(teams=[0, 1, 0, 1, 2, 2])])
def test_env_builds_n_player_boards(kw):
    env = GeneralsEnv(grid_dims=(12, 12), truncation=10, pool_size=3, min_generals_distance=4, **kw)
    n = env.num_players
    pool, state = env.reset(jrandom.PRNGKey(3))
    assert pool.ownership.shape == (3, n, 12, 12)
    for k in range(3):
        s = jax.tree.map(lambda x: x[k], pool)
        assert int(s.generals.sum()) == n
        assert s.general_positions.min() >= 0
        assert [int(s.ownership[i].sum()) for i in range(n)] == [1] * n
        # castles are read above the general range, never as generals
        assert not bool((s.castles & s.generals).any())
    ts, _ = env.step(state, passes(n), pool)
    assert ts.reward.shape == (n,) and ts.observation.owned_cells.shape == (n, 12, 12)


def test_env_rejects_inconsistent_player_settings():
    with pytest.raises(ValueError):
        GeneralsEnv(num_players=3, teams=[0, 0, 1, 1])
    with pytest.raises(NotImplementedError):
        GeneralsEnv(num_players=4, deathtouch_turn=800)


def test_env_rewards_follow_the_team_result():
    env = GeneralsEnv(grid_dims=(6, 6), teams=TEAMS_2V2, truncation=100, pool_size=2)
    pool, _ = env.reset(jrandom.PRNGKey(0))
    s = board({0: (0, 0), 1: (5, 5), 2: (0, 1), 3: (0, 2)}, teams=TEAMS_2V2)
    s = s._replace(armies=s.armies.at[0, 0].set(20))

    ts, s = env.step(s, jnp.stack([move(0, 0, RIGHT), PASS, PASS, PASS]), pool)
    assert ts.reward.tolist() == [0.0, 0.0, 0.0, 0.0]          # game still on
    assert not bool(ts.terminated)

    s = s._replace(armies=s.armies.at[0, 1].set(20))
    ts, s = env.step(s, jnp.stack([move(0, 1, RIGHT), PASS, PASS, PASS]), pool)
    assert ts.reward.tolist() == [1.0, 1.0, -1.0, -1.0]
    assert bool(ts.terminated) and not bool(ts.truncated)
    assert int(ts.info.winner) == 0
    assert bool(ts.last_state.eliminated[2]) and bool(ts.last_state.eliminated[3])
    # auto-reset handed out a fresh pool board
    assert int(s.time) == 0 and not bool(s.eliminated.any()) and int(s.winner) == -1


def test_env_truncation_in_multiplayer():
    env = GeneralsEnv(grid_dims=(8, 8), num_players=3, truncation=5, pool_size=2, min_generals_distance=2)
    pool, s = env.reset(jrandom.PRNGKey(1))
    for t in range(1, 5):
        ts, s = env.step(s, passes(3), pool)
        assert not bool(ts.truncated) and int(s.time) == t
    ts, s = env.step(s, passes(3), pool)
    assert bool(ts.truncated) and not bool(ts.terminated)
    assert ts.reward.tolist() == [0.0, 0.0, 0.0]
    assert int(s.time) == 0                                    # reset from the pool


def test_env_reward_in_two_player_game_is_the_old_one():
    env = GeneralsEnv(grid_dims=(6, 6), truncation=100, pool_size=2)
    pool, _ = env.reset(jrandom.PRNGKey(0))
    s = board({0: (0, 0), 1: (0, 1)}, size=6)
    s = s._replace(armies=s.armies.at[0, 1].set(20))
    ts, _ = env.step(s, jnp.stack([PASS, move(0, 1, LEFT)]), pool)
    assert ts.reward.tolist() == [-1.0, 1.0] and int(ts.info.winner) == 1


def test_build_castles_works_for_every_player():
    s = board({0: (0, 0), 1: (0, 7), 2: (7, 0), 3: (7, 7)}, size=8, teams=FFA4)
    # both sites are 7+ steps from their builder's general: base price 35
    s = give(s, 2, (3, 3), 60)
    s = give(s, 3, (2, 2), 60)
    build = lambda i, j: jnp.array([bc.BUILD, i, j, 0, 0], dtype=jnp.int32)
    ns, _ = bc.step(s, jnp.stack([PASS, PASS, build(3, 3), build(2, 2)]))
    assert bool(ns.castles[3, 3]) and bool(ns.castles[2, 2])
    assert int(ns.armies[3, 3]) == 60 - 35 and int(ns.armies[2, 2]) == 60 - 35
    assert bool(ns.ownership[2, 3, 3]) and bool(ns.ownership[3, 2, 2])


# -------------------------------------------------------- it actually runs


@pytest.mark.parametrize("teams", [None, [0, 0, 1, 1], [0, 1, 2, 3]])
def test_batched_random_play_under_jit(teams):
    """A vmapped batch plays 200 turns of random valid moves under jax.jit and
    stays well-formed: ownership planes are disjoint, eliminated players own
    nothing, a finished game names a team, and the winner's team is +1."""
    kw = dict(teams=teams) if teams is not None else {}
    env = GeneralsEnv(grid_dims=(9, 9), truncation=120, pool_size=16, min_generals_distance=3, **kw)
    n = env.num_players
    num_envs = 16
    pool, _ = env.reset(jrandom.PRNGKey(0))
    states = jax.tree.map(lambda x: x[:num_envs], pool)
    states = states._replace(pool_idx=jnp.arange(num_envs, dtype=jnp.int32))

    def act(state, keys):
        return jnp.stack([sample_valid_action(keys[i], game.get_observation(state, i)) for i in range(n)])

    @jax.jit
    def run(states, key):
        def body(carry, _):
            states, key = carry
            key, k = jrandom.split(key)
            keys = jrandom.split(k, num_envs * n).reshape(num_envs, n, -1)
            actions = jax.vmap(act)(states, keys)
            ts, states = jax.vmap(env.step, in_axes=(0, 0, None))(states, actions, pool)
            last = ts.last_state
            disjoint = jnp.all(jnp.sum(last.ownership, axis=1) <= 1, axis=(1, 2))
            elim_own_nothing = jnp.all(~last.eliminated | (jnp.sum(last.ownership, axis=(2, 3)) == 0), axis=1)
            team_reward_ok = jnp.all(
                jnp.where(ts.info.winner[:, None] >= 0,
                          ts.reward == jnp.where(env.teams[None] == ts.info.winner[:, None], 1.0, -1.0),
                          ts.reward == 0.0), axis=1)
            finished = ts.terminated | ts.truncated
            return (states, key), (disjoint & elim_own_nothing & team_reward_ok, ts.terminated, ts.info.winner)
        _, (ok, terminated, winner) = jax.lax.scan(body, (states, key), None, length=200)
        return ok, terminated, winner

    ok, terminated, winner = run(states, jrandom.PRNGKey(1))
    assert bool(ok.all())
    assert bool(jnp.all(winner[terminated] >= 0))
    assert bool(jnp.all(winner[~terminated] == -1))
