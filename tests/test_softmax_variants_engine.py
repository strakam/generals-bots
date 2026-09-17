"""Hosted mode boundaries: four distinct seats and real castle construction."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from pydantic import ValidationError

from generals import GeneralsEnv
from generals.core import game
from generals.core.match import make_board
from integrations.softmax.config import GameConfig
from integrations.softmax.engine import Match
from integrations.softmax.protocol import PASS, parse_action


@pytest.fixture(scope="module")
def engines():
    return {"classic": Match(7), "ffa": Match(7, 4), "build": Match(7, ruleset="build_castles")}


@pytest.fixture
def matches(engines):
    saved = {name: (match.state, match.height, match.width) for name, match in engines.items()}
    yield engines
    for name, match in engines.items():
        match.state, match.height, match.width = saved[name]
        match.last_move_executed = [None] * match.num_players
        match.last_build_executed = [None] * match.num_players


def board(match, grid):
    match.state = game.create_initial_state(jnp.asarray(grid, dtype=jnp.int32), num_players=match.num_players)
    match.height, match.width = match.state.armies.shape


def test_config_accepts_supported_rosters_and_rejects_ambiguous_seats():
    classic = {"tokens": ["a", "b"], "players": [{"name": "Red"}, {"name": "Blue"}]}
    assert GameConfig(**classic).ruleset == "classic"
    assert GameConfig(**classic, ruleset="build_castles").max_turns == 1200
    ffa = {"tokens": ["a", "b", "c", "d"], "players": [{"name": str(n)} for n in range(4)]}
    assert len(GameConfig(**ffa).players) == 4
    for invalid in (
        {**ffa, "tokens": ["a", "b", "c", "c"]},
        {**ffa, "tokens": ["a", "b"]},
        {"tokens": ["a", "b", "c"], "players": ffa["players"][:3]},
        {**ffa, "ruleset": "build_castles"},
        {**classic, "ruleset": "deathtouch"},
    ):
        with pytest.raises(ValidationError):
            GameConfig(**invalid)


def test_generated_modes_spawn_all_players_without_changing_classic(matches):
    for name, match in matches.items():
        n = 4 if name == "ffa" else 2
        assert match.state.ownership.shape == (n, match.height, match.width)
        assert match.active_slots == list(range(n))
        assert int(match.state.generals.sum()) == n
        assert match.env.deathtouch_turn is None
        assert 18 <= match.height <= 21 and 18 <= match.width <= 21
        assert np.all(np.asarray(match.state.ownership).sum(axis=(1, 2)) == 1)
        assert bool(match.state.castles.any()) == (name != "build")
    original_env = GeneralsEnv(min_grid_size=18, max_grid_size=21, pad_to=21, truncation=1200,
                              mountain_density_range=(0.24, 0.26), min_generals_distance=17)
    original = make_board(original_env, 7)
    for before, after in zip(jax.tree.leaves(original), jax.tree.leaves(matches["classic"].state)):
        np.testing.assert_array_equal(before, after)


def test_variable_board_preserves_team_assignments():
    env = GeneralsEnv(min_grid_size=18, max_grid_size=21, pad_to=21, teams=[0, 0, 1, 1],
                      min_generals_distance=10, build_castles=True)
    state = make_board(env, 7)
    assert state.teams.tolist() == [0, 0, 1, 1]
    assert int(state.generals.sum()) == 4
    assert not bool(state.castles.any())


def test_ffa_identities_are_visible_only_through_fog(matches):
    match = matches["ffa"]
    grid = np.zeros((8, 8), dtype=np.int32)
    grid[0, 0], grid[0, 1], grid[7, 6], grid[7, 7] = 1, 2, 3, 4
    board(match, grid)
    obs = match.observation(0)
    assert obs["owner_grid"][0][1] == 2  # Existing bots keep relative enemy ownership.
    assert obs["visible_owner_grid"][0][1] == 2
    assert obs["visible_owner_grid"][7][6] == 0
    assert obs["visible_owner_grid"][7][7] == 0
    assert obs["army_grid"][7][7] == 0
    assert obs["opp_land"] == 3 and obs["opp_army"] == 3
    assert "build_cost_grid" not in obs
    third = match.observation(2)
    assert third["owner_grid"][7][7] == 2
    assert third["visible_owner_grid"][7][7] == 4
    assert match.frame()["owner_grid"][7][7] == 4


def test_ffa_capture_eliminates_one_player_but_match_continues(matches):
    match = matches["ffa"]
    board(match, [[1, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [3, 0, 0, 4]])
    match.state = match.state._replace(armies=match.state.armies.at[0, 0].set(10))
    assert match.advance([[0, 0, 0, 3, 0], PASS, PASS, PASS]) == -1
    assert match.active_slots == [0, 2, 3]
    assert match.observation(1)["eliminated"] is True
    assert np.asarray(match.observation(1)["visible_owner_grid"]).sum() == 0
    assert match.frame()["eliminated"] == [False, True, False, False]
    assert match.last_move_executed == [True, None, None, None]
    assert match.forfeit([2, 3]) == 0
    assert match.active_slots == [0]


def test_ffa_forfeit_neutralizes_territory_and_preserves_remaining_players(matches):
    match = matches["ffa"]
    board(match, [[1, 2, -2, 0], [0, 0, 0, 0], [0, 0, 40, 0], [3, 0, 0, 4]])
    match.state = match.state._replace(armies=match.state.armies.at[0, 1].set(25))
    original_armies = np.asarray(match.state.armies).copy()
    assert match.forfeit([1]) == -1
    assert match.active_slots == [0, 2, 3]
    assert bool(match.state.ownership_neutral[0, 1])
    assert bool(match.state.castles[0, 1]) and not bool(match.state.generals[0, 1])
    assert bool(match.state.castles[2, 2]) and bool(match.state.mountains[0, 2])
    np.testing.assert_array_equal(match.state.armies, original_armies)
    assert match.advance([PASS] * 4) == -1
    assert match.forfeit([0, 2, 3]) == -1
    assert match.active_slots == []


def test_castle_build_price_action_and_receipt_match_engine(matches):
    match = matches["build"]
    board(match, [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 2]])
    state = match.state
    match.state = state._replace(
        ownership=state.ownership.at[0, 0, 1].set(True),
        ownership_neutral=state.ownership_neutral.at[0, 1].set(False),
        armies=state.armies.at[0, 1].set(50),
    )
    obs = match.observation(0)
    assert obs["build_cost_grid"][0][1] == 47
    assert sum(map(sum, obs["build_cost_grid"])) == 47
    action = [2, 0, 1, 0, 0]
    message = {"type": "action", "turn": 0, "action": action}
    with pytest.raises(ValueError):
        parse_action(message, 0, 4, 4)
    assert parse_action(message, 0, 4, 4, ruleset="build_castles") == action
    assert match.advance([action, action]) == -1  # An enemy trying the same cell must get a failure receipt.
    assert bool(match.state.castles[0, 1])
    assert int(match.state.armies[0, 1]) == 3
    assert match.last_build_executed == [True, False]
    assert match.last_move_executed == [None, None]
    assert match.observation(0)["build_cost_grid"][0][1] == 0
    match.advance([action, PASS])  # Existing castles cannot be built twice.
    assert match.last_build_executed == [False, None]
    assert int(match.state.armies[0, 1]) == 4  # Only normal even-tick growth.


def test_build_rejects_unowned_general_and_unaffordable_cells(matches):
    match = matches["build"]
    board(match, [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 2]])
    for row, col in ((0, 0), (0, 1), (3, 3)):
        match.advance([[2, row, col, 0, 0], PASS])
        assert match.last_build_executed == [False, None]
        assert not bool(match.state.castles.any())
    state = match.state
    match.state = state._replace(
        ownership=state.ownership.at[0, 0, 1].set(True),
        ownership_neutral=state.ownership_neutral.at[0, 1].set(False),
        armies=state.armies.at[0, 1].set(46),
    )
    match.advance([[2, 0, 1, 0, 0], PASS])
    assert match.last_build_executed == [False, None]
    assert not bool(match.state.castles.any())


def test_build_receipt_survives_same_turn_capture_and_move_uses_post_build_army(matches):
    match = matches["build"]
    board(match, [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 2]])
    state = match.state
    match.state = state._replace(
        ownership=state.ownership.at[0, 0, 1].set(True).at[1, 0, 2].set(True),
        ownership_neutral=state.ownership_neutral.at[0, 1:3].set(False),
        armies=state.armies.at[0, 1].set(50).at[0, 2].set(10),
    )
    match.advance([[2, 0, 1, 0, 0], [0, 0, 2, 2, 0]])
    assert match.last_build_executed == [True, None]
    assert match.last_move_executed == [None, True]
    assert bool(match.state.castles[0, 1])
    assert bool(match.state.ownership[1, 0, 1])
    assert int(match.state.armies[0, 1]) == 6  # Nine attackers beat the three left after building.
