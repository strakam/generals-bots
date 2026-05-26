"""Tests for JAX-based game implementation."""
import jax
import jax.numpy as jnp
import pytest

from generals.core import game


TEAMS_2P = jnp.arange(2, dtype=jnp.int32)


def create_test_grid(size=4):
    """Create a simple test grid with generals in corners."""
    grid = jnp.zeros((size, size), dtype=jnp.int32)
    grid = grid.at[0, 0].set(1)  # General player 0
    grid = grid.at[size - 1, size - 1].set(2)  # General player 1
    return grid


def test_create_initial_state():
    """Test creating initial state from a grid."""
    grid = create_test_grid(4)
    state = game.create_initial_state(grid, TEAMS_2P)

    # Check state structure
    assert hasattr(state, "armies")
    assert hasattr(state, "ownership")
    assert hasattr(state, "generals")
    assert hasattr(state, "time")
    assert hasattr(state, "winner")

    # Check initial armies
    assert state.armies[0, 0] == 1  # General A
    assert state.armies[3, 3] == 1  # General B

    # Check ownership
    assert state.ownership[0, 0, 0] == True  # Player 0 owns (0,0)
    assert state.ownership[1, 3, 3] == True  # Player 1 owns (3,3)

    # Check initial game state
    assert state.time == 0
    assert state.winner == -1


def test_step_pass_action():
    """Test that pass actions don't change state."""
    grid = create_test_grid(2)
    state = game.create_initial_state(grid, TEAMS_2P)

    # Both players pass
    actions = jnp.array(
        [
            [1, 0, 0, 0, 0],  # Player 0 passes
            [1, 0, 0, 0, 0],  # Player 1 passes
        ],
        dtype=jnp.int32,
    )

    new_state, info = game.step(state, actions)

    # Armies should not change (except time increment)
    assert jnp.array_equal(new_state.armies, state.armies)
    assert new_state.time == 1
    assert new_state.winner == -1


def test_step_move_to_neutral():
    """Test moving to a neutral cell."""
    grid = create_test_grid(3)
    state = game.create_initial_state(grid, TEAMS_2P)

    # Give player 0 more armies
    state = state._replace(armies=state.armies.at[0, 0].set(5))

    # Player 0 moves right (direction 3)
    actions = jnp.array(
        [
            [0, 0, 0, 3, 0],  # Move right from (0,0)
            [1, 0, 0, 0, 0],  # Player 1 passes
        ],
        dtype=jnp.int32,
    )

    new_state, info = game.step(state, actions)

    # Check armies moved
    assert new_state.armies[0, 0] == 1  # Left 1 behind
    assert new_state.armies[0, 1] == 4  # Moved 4

    # Check ownership changed
    assert new_state.ownership[0, 0, 1] == True


def test_step_move_to_own_cell():
    """Test moving to own cell (merge armies)."""
    grid = create_test_grid(3)
    state = game.create_initial_state(grid, TEAMS_2P)

    # Setup: Give player 0 two cells with armies
    state = state._replace(
        armies=state.armies.at[0, 0].set(5).at[0, 1].set(3),
        ownership=state.ownership.at[0, 0, 1].set(True),
        ownership_neutral=state.ownership_neutral.at[0, 1].set(False),
    )

    # Player 0 moves from (0,0) to (0,1)
    actions = jnp.array(
        [
            [0, 0, 0, 3, 0],  # Move right
            [1, 0, 0, 0, 0],  # Pass
        ],
        dtype=jnp.int32,
    )

    new_state, info = game.step(state, actions)

    # Armies should merge
    assert new_state.armies[0, 0] == 1  # Left 1 behind
    assert new_state.armies[0, 1] == 7  # 3 + 4 moved


def test_get_observation():
    """Test observation generation with fog of war."""
    grid = create_test_grid(4)
    state = game.create_initial_state(grid, TEAMS_2P)

    obs = game.get_observation(state, 0)

    # Check observation structure
    assert hasattr(obs, "armies")
    assert hasattr(obs, "owned_cells")
    assert hasattr(obs, "fog_cells")
    assert hasattr(obs, "timestep")

    # Player 0 should see their general
    assert obs.armies[0, 0] == 1

    # Player 0 should not see player 1's general (too far)
    assert obs.armies[3, 3] == 0  # Hidden in fog


def test_global_update():
    """Test global army increment mechanics."""
    grid = create_test_grid(2)
    state = game.create_initial_state(grid, TEAMS_2P)
    state = state._replace(armies=state.armies.at[0, 0].set(5), time=jnp.int32(2))
    state = game.global_update(state)

    # General should have gained 1 army
    assert state.armies[0, 0] == 6


def test_batch_step():
    """Test batched step execution."""
    grid = create_test_grid(2)
    state = game.create_initial_state(grid, TEAMS_2P)

    # Stack into batch
    batched_state = jax.tree.map(lambda x: jnp.stack([x, x]), state)

    # Create actions for both envs
    actions = jnp.array(
        [
            [[1, 0, 0, 0, 0], [1, 0, 0, 0, 0]],  # Env 0: both pass
            [[1, 0, 0, 0, 0], [1, 0, 0, 0, 0]],  # Env 1: both pass
        ],
        dtype=jnp.int32,
    )

    new_states, infos = game.batch_step(batched_state, actions)

    # Check batch dimension preserved
    assert new_states.time.shape == (2,)
    assert new_states.armies.shape == (2, 2, 2)


def test_jit_compilation():
    """Test that step function can be JIT compiled."""
    grid = create_test_grid(2)
    state = game.create_initial_state(grid, TEAMS_2P)

    # JIT compile step
    jitted_step = jax.jit(game.step)

    actions = jnp.array(
        [
            [1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
        ],
        dtype=jnp.int32,
    )

    # Should execute without errors
    new_state, info = jitted_step(state, actions)

    assert new_state.time == 1


def test_multiplayer_state_shapes():
    """N=4 FFA: ownership has 4 player planes, info arrays are length 4."""
    teams = jnp.arange(4, dtype=jnp.int32)
    grid = jnp.zeros((8, 8), dtype=jnp.int32)
    grid = grid.at[0, 0].set(1)
    grid = grid.at[0, 7].set(2)
    grid = grid.at[7, 0].set(3)
    grid = grid.at[7, 7].set(4)
    state = game.create_initial_state(grid, teams)

    assert state.ownership.shape == (4, 8, 8)
    assert state.general_positions.shape == (4, 2)
    assert state.teams.shape == (4,)
    assert state.eliminated.shape == (4,)
    assert bool(jnp.all(~state.eliminated))

    info = game.get_info(state)
    assert info.army.shape == (4,)
    assert info.land.shape == (4,)


def test_teammate_merge_transfers_ownership():
    """2v2: when P0 walks onto P1's cell, armies merge and ownership transfers."""
    teams = jnp.array([0, 0, 1, 1], dtype=jnp.int32)
    grid = jnp.zeros((4, 4), dtype=jnp.int32)
    grid = grid.at[0, 0].set(1)
    grid = grid.at[0, 3].set(2)
    grid = grid.at[3, 0].set(3)
    grid = grid.at[3, 3].set(4)
    state = game.create_initial_state(grid, teams)
    state = state._replace(
        armies=state.armies.at[0, 0].set(5).at[0, 1].set(3),
        ownership=state.ownership.at[1, 0, 1].set(True),
        ownership_neutral=state.ownership_neutral.at[0, 1].set(False),
    )

    actions = jnp.array([[0, 0, 0, 3, 0], [1, 0, 0, 0, 0], [1, 0, 0, 0, 0], [1, 0, 0, 0, 0]], dtype=jnp.int32)
    new_state, _ = game.step(state, actions)

    # P0 moves 4 armies into P1's cell (3 + 4 = 7). Ownership transfers to P0.
    assert int(new_state.armies[0, 1]) == 7
    assert bool(new_state.ownership[0, 0, 1])
    assert not bool(new_state.ownership[1, 0, 1])


def test_general_capture_halves_armies_and_converts_to_city():
    """3-player FFA: capturer takes all of captured player's cells with armies halved; general → city."""
    teams = jnp.arange(3, dtype=jnp.int32)
    grid = jnp.zeros((5, 5), dtype=jnp.int32)
    grid = grid.at[0, 0].set(1)
    grid = grid.at[0, 1].set(2)
    grid = grid.at[4, 4].set(3)
    state = game.create_initial_state(grid, teams)
    state = state._replace(
        armies=state.armies.at[0, 0].set(50).at[0, 1].set(3).at[0, 2].set(20),
        ownership=state.ownership.at[1, 0, 2].set(True),
        ownership_neutral=state.ownership_neutral.at[0, 2].set(False),
    )

    actions = jnp.array([[0, 0, 0, 3, 0], [1, 0, 0, 0, 0], [1, 0, 0, 0, 0]], dtype=jnp.int32)
    new_state, info = game.step(state, actions)

    # P1's general at (0,1) had army=3; halved → 1; +1 from city structure increment at t=1 → 2
    assert int(new_state.armies[0, 1]) == 2
    # P1's other cell at (0,2) had army=20; halved → 10; not a structure, no increment
    assert int(new_state.armies[0, 2]) == 10
    assert bool(new_state.ownership[0, 0, 1])
    assert bool(new_state.ownership[0, 0, 2])
    assert not bool(jnp.any(new_state.ownership[1]))
    assert not bool(new_state.generals[0, 1])
    assert bool(new_state.cities[0, 1])
    assert bool(new_state.eliminated[1])
    # Game continues — P2 still alive
    assert int(info.winner) == -1
    assert not bool(info.is_done)


def test_last_team_standing_wins_in_2v2():
    """2v2: winner is set only after both enemy team members are eliminated."""
    teams = jnp.array([0, 0, 1, 1], dtype=jnp.int32)
    grid = jnp.zeros((6, 6), dtype=jnp.int32)
    grid = grid.at[0, 0].set(1)
    grid = grid.at[5, 5].set(2)
    grid = grid.at[0, 1].set(3)
    grid = grid.at[0, 2].set(4)
    state = game.create_initial_state(grid, teams)
    state = state._replace(armies=state.armies.at[0, 0].set(20))

    # P0 captures P2
    actions = jnp.array([[0, 0, 0, 3, 0], [1, 0, 0, 0, 0], [1, 0, 0, 0, 0], [1, 0, 0, 0, 0]], dtype=jnp.int32)
    state, info = game.step(state, actions)
    assert bool(state.eliminated[2])
    assert int(info.winner) == -1  # P3 still alive

    # Boost P0 and capture P3
    state = state._replace(armies=state.armies.at[0, 1].set(20))
    actions = jnp.array([[0, 0, 1, 3, 0], [1, 0, 0, 0, 0], [1, 0, 0, 0, 0], [1, 0, 0, 0, 0]], dtype=jnp.int32)
    state, info = game.step(state, actions)
    assert bool(state.eliminated[3])
    assert int(info.winner) == 0
    assert bool(info.is_done)


def test_team_shared_visibility():
    """2v2: visibility extends to 3x3 around any teammate's cell."""
    teams = jnp.array([0, 0, 1, 1], dtype=jnp.int32)
    grid = jnp.zeros((6, 6), dtype=jnp.int32)
    grid = grid.at[0, 0].set(1)  # P0
    grid = grid.at[0, 5].set(2)  # P1 (teammate)
    grid = grid.at[5, 0].set(3)
    grid = grid.at[5, 5].set(4)
    state = game.create_initial_state(grid, teams)

    obs = game.get_observation(state, 0)
    # Cell adjacent to P1's general is visible to P0 (team-shared sight)
    assert not bool(obs.fog_cells[0, 4])
    assert bool(obs.allied_cells[0, 5])
    # Middle of the board is in fog
    assert bool(obs.fog_cells[3, 3])


def test_env_multiplayer_smoke():
    """End-to-end: N=4 FFA env runs without crashing."""
    from generals import GeneralsEnv
    import jax.random as jrandom

    env = GeneralsEnv(grid_dims=(10, 10), num_players=4, truncation=20, pool_size=4)
    pool, state = env.reset(jrandom.PRNGKey(0))
    actions = jnp.array([[1, 0, 0, 0, 0]] * 4, dtype=jnp.int32)
    ts, state = env.step(state, actions, pool)
    assert ts.reward.shape == (4,)
    assert ts.observation.owned_cells.shape == (4, 10, 10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
