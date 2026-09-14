"""Coworld rule parity, player boundaries, and complete episode lifecycle."""

import json

import jax.numpy as jnp
import pytest

pytest.importorskip("fastapi", reason="install .[softmax] for integration tests")
pytest.importorskip("httpx", reason="install .[softmax] for integration tests")
from fastapi.testclient import TestClient
from pydantic import ValidationError
from starlette.websockets import WebSocketDisconnect

from generals.core.game import create_initial_state
from integrations.softmax.artifacts import read_json, write_json
from integrations.softmax.config import GameConfig
from integrations.softmax.engine import Match, executed_moves
from integrations.softmax.protocol import PASS, parse_action, stdio_frame
from integrations.softmax.server import create_app


def config(**overrides):
    return GameConfig.model_validate(
        {
            "tokens": ["red-secret", "blue-secret"],
            "players": [{"name": "Red"}, {"name": "Blue"}],
            "seed": 7,
            "max_turns": 4,
            **overrides,
        }
    )


def client_for(tmp_path, **overrides):
    artifacts = {name: (tmp_path / f"{name}.json").as_uri() for name in ("results", "replay", "failure")}
    return TestClient(create_app(config(**overrides), artifacts))


def connect(client, slot):
    token = ["red-secret", "blue-secret"][slot]
    return client.websocket_connect(f"/player?slot={slot}&token={token}")


def receive(ws, kind):
    while True:
        message = ws.receive_json()
        if message["type"] == kind:
            return message


def act(ws, observation, action=PASS):
    ws.send_json({"type": "action", "turn": observation["turn"], "action": action})


@pytest.mark.parametrize(
    "action",
    [
        [],
        [0, 0, 0, 0],
        [0, -1, 0, 0, 0],
        [0, 21, 0, 0, 0],
        [0, 0, 0, -1, 0],
        [0, 0, 0, 4, 0],
        [0, 0, 0, 0, 2],
        [3, 0, 0, 0, 0],
        [2, 0, 0, 0, 0],
        [False, 0, 0, 0, 0],
        [0, 2**100, 0, 0, 0],
        [0, 0.0, 0, 0, 0],
        "1 0 0 0 0",
    ],
)
def test_untrusted_actions_rejected(action):
    with pytest.raises(ValueError):
        parse_action({"type": "action", "turn": 8, "action": action}, 8, 21, 21)


@pytest.mark.parametrize("turn", [7, 9, True, "8"])
def test_action_turn_is_exact(turn):
    with pytest.raises(ValueError):
        parse_action({"type": "action", "turn": turn, "action": PASS}, 8, 21, 21)


@pytest.mark.parametrize(
    "fields",
    [
        {"tokens": ["same", "same"]},
        {"max_turns": 1201},
        {"seed": -1},
        {"perfect_info": True},
        {"seed": True},
        {"tick_interval_seconds": 1},
        {"turn_timeout_seconds": 20},
    ],
)
def test_config_cannot_change_rules_or_overrun_hosted_deadline(fields):
    with pytest.raises(ValidationError):
        config(**fields)


def test_random_seed_is_default():
    assert GameConfig(tokens=["a", "b"], players=[{"name": "A"}, {"name": "B"}]).seed is None


def test_classic_rules_have_neutral_castles_and_reject_builds():
    match = Match(7)
    castles = match.state.castles
    assert 9 <= int(castles.sum()) <= 11
    assert bool(jnp.all(match.state.ownership_neutral[castles]))
    assert bool(jnp.all((match.state.armies[castles] >= 40) & (match.state.armies[castles] <= 50)))
    with pytest.raises(ValueError, match="moves and passes"):
        match.advance([[2, 0, 0, 0, 0], PASS])


def test_classic_castles_can_be_captured_and_generate_armies():
    match = Match(7)
    grid = jnp.zeros((8, 8), dtype=jnp.int32).at[0, 0].set(1).at[7, 7].set(2).at[0, 1].set(40)
    state = create_initial_state(grid)
    match.state = state._replace(armies=state.armies.at[0, 0].set(50))
    match.advance([[0, 0, 0, 3, 0], PASS])
    assert bool(match.state.castles[0, 1]) and bool(match.state.ownership[0, 0, 1])
    army = int(match.state.armies[0, 1])
    assert army == 9  # 49 attackers minus the 40 neutral defenders.
    match.advance([PASS, PASS])
    assert int(match.state.armies[0, 1]) == army + 1


def test_engine_observation_matches_existing_stdio_protocol():
    from competition.protocol import encode_observation
    from generals.core.game import get_observation

    match = Match(7)
    for _ in range(3):
        for slot in range(2):
            assert stdio_frame(match.observation(slot)) == encode_observation(get_observation(match.state, slot))
            assert match.observation(slot)["last_move_executed"] is None
        match.advance([PASS, PASS])


@pytest.mark.parametrize("attacking_army,expected", [(10, [False, True]), (8, [True, True])])
def test_move_receipts_follow_actual_execution_order(attacking_army, expected):
    state = create_initial_state(jnp.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 2]]))
    state = state._replace(
        ownership=state.ownership.at[0, 1, 1:3].set(True).at[1, 1, 0].set(True),
        ownership_neutral=state.ownership_neutral.at[1, :3].set(False),
        armies=state.armies.at[1, 0].set(attacking_army).at[1, 1].set(10).at[1, 2].set(5),
    )
    # Blue's chase resolves first. Red may still own BOTH endpoints but be
    # reduced to one army, making its move a no-op invisible to ownership checks.
    actions = jnp.array([[0, 1, 1, 3, 0], [0, 1, 0, 3, 0]])
    assert executed_moves(state, actions).tolist() == expected


def test_episode_records_reproducible_replay_and_scores(tmp_path):
    with client_for(tmp_path) as client, connect(client, 0) as red, connect(client, 1) as blue:
        for _ in range(4):
            obs = [receive(ws, "observation") for ws in (red, blue)]
            for ws, observation in zip((red, blue), obs):
                act(ws, observation)
        result = receive(red, "final")["result"]
        assert receive(blue, "final")["result"] == result
        assert result["scores"] == [0, 0] and result["reason"] == "turn_limit"
        assert result["timeouts"] == [0, 0]
        replay = client.get("/replay.json").json()
        assert len(replay["frames"]) == 5 and len(replay["turns"]) == 4
        assert replay == json.loads((tmp_path / "replay.json").read_text())
        assert not (tmp_path / "failure.json").exists()
    match = Match(replay["seed"])
    assert replay["frames"][0] == match.frame()
    for turn, frame in zip(replay["turns"], replay["frames"][1:]):
        assert turn["applied"]
        match.advance(turn["actions"])
        assert frame == match.frame()


def test_live_public_routes_do_not_reveal_hidden_state(tmp_path):
    with client_for(tmp_path) as client:
        assert client.get("/healthz").status_code == 200
        assert client.get("/replay.json").status_code == 409
        assert client.get("/client/replay.json").status_code == 409
        for asset in ("app.js", "board.css", "assets/crownie.png", "fonts/Quicksand-VariableFont_wght.ttf"):
            assert client.get("/client/static/" + asset).status_code == 200
        assert client.get("/config.json").status_code == 404
        assert client.get("/client/player?slot=0&token=wrong").status_code == 403
        page = client.get("/client/player?slot=0&token=red-secret")
        assert page.status_code == 200 and page.headers["referrer-policy"] == "no-referrer"
        hosted_address = "wss://example.com/session/proxy/player?slot=0&token=red-secret"
        page = client.get("/client/player", params={"address": hosted_address})
        assert page.status_code == 200 and page.headers["referrer-policy"] == "no-referrer"
        for address in (
            hosted_address.replace("red-secret", "wrong"),
            hosted_address.replace("wss:", "https:"),
            hosted_address + "&slot=1",
            hosted_address.replace("slot=0&", ""),
            "wss://[broken",
            "",
        ):
            # Valid legacy parameters must not override an invalid hosted address.
            assert client.get("/client/player", params={
                "address": address, "slot": "0", "token": "red-secret",
            }).status_code == 403
        with client.websocket_connect("/global") as global_ws:
            message = global_ws.receive_json()
            assert message["board"] is None
            assert set(message) == {
                "type",
                "protocol_version",
                "phase",
                "turn",
                "max_turns",
                "players",
                "army",
                "land",
                "result",
                "board",
            }
            assert "secret" not in json.dumps(message)
            assert "seed" not in json.dumps(message)
        with connect(client, 0) as red, connect(client, 1) as blue:
            obs = receive(red, "observation")
            receive(blue, "observation")
            assert "seed" not in obs and "tokens" not in obs
            # At spawn the enemy general's location and all fog armies remain hidden.
            assert sum(row.count(4) for row in obs["type_grid"]) == 1
            for types, armies, owners in zip(obs["type_grid"], obs["army_grid"], obs["owner_grid"]):
                for kind, army, owner in zip(types, armies, owners):
                    if kind in (0, 5):
                        assert army == 0 and owner == 0
            assert client.get("/replay.json").status_code == 409


def test_wrong_and_duplicate_player_connections_are_rejected(tmp_path):
    with client_for(tmp_path) as client:
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect("/player?slot=1&token=red-secret"):
                pass
        with connect(client, 0) as red:
            assert receive(red, "hello")["slot"] == 0
            with pytest.raises(WebSocketDisconnect):
                with connect(client, 0):
                    pass


def test_duplicate_and_stale_actions_do_not_replace_first_action(tmp_path):
    with client_for(tmp_path, max_turns=1) as client, connect(client, 0) as red, connect(client, 1) as blue:
        robs, bobs = receive(red, "observation"), receive(blue, "observation")
        red.send_json({"type": "action", "turn": 900, "action": PASS})
        assert "current turn" in receive(red, "error")["message"]
        act(red, robs)
        act(red, robs, [0, 0, 0, 0, 0])
        assert "first valid" in receive(red, "error")["message"]
        act(blue, bobs)
        receive(red, "final")
        assert client.get("/replay.json").json()["turns"][0]["actions"][0] == PASS


def test_build_rejection_does_not_consume_the_action_slot(tmp_path):
    with client_for(tmp_path, max_turns=1) as client, connect(client, 0) as red, connect(client, 1) as blue:
        robs, bobs = receive(red, "observation"), receive(blue, "observation")
        act(red, robs, [2, 0, 0, 0, 0])
        assert "invalid action kind" in receive(red, "error")["message"]
        act(red, robs)
        act(blue, bobs)
        assert receive(red, "final")["result"]["timeouts"] == [0, 0]


def test_silent_player_forfeits_without_hanging_episode(tmp_path):
    with client_for(tmp_path, max_consecutive_timeouts=2, turn_timeout_seconds=0.1) as client:
        with connect(client, 0) as red, connect(client, 1) as blue:
            for _ in range(2):
                act(red, receive(red, "observation"))
                receive(blue, "observation")  # deliberately no reply
            result = receive(red, "final")["result"]
            assert result["winner"] == 0 and result["scores"] == [1, -1]
            assert result["reason"] == "forfeit" and result["timeouts"] == [0, 2]
            replay = client.get("/replay.json").json()
            assert replay["turns"][-1]["applied"] is False


def test_missing_player_writes_typed_failure_without_success(tmp_path):
    with client_for(tmp_path, player_connect_timeout_seconds=1.0) as client, connect(client, 0) as red:
        receive(red, "failure")
        failure = json.loads((tmp_path / "failure.json").read_text())
        assert failure["failed_policy_index"] == 1
        assert set(failure) == {"message", "failed_policy_index"}
        assert not (tmp_path / "results.json").exists()
        assert not (tmp_path / "replay.json").exists()


@pytest.mark.parametrize("attack, scores, reason", [(3, [0, 0], "turn_limit"), (502, [1, -1], "general_capture")])
def test_general_capture_requires_winning_combat_after_turn_800(tmp_path, attack, scores, reason):
    with client_for(tmp_path, max_turns=802) as client:
        episode = client.app.state.episode
        grid = jnp.zeros((8, 8), dtype=jnp.int32).at[0, 0].set(1).at[0, 7].set(2)
        state = create_initial_state(grid)
        episode.match.state = state._replace(
            time=jnp.int32(801),
            armies=state.armies.at[0, 6].set(attack).at[0, 7].set(500),
            ownership=state.ownership.at[0, 0, 6].set(True),
            ownership_neutral=state.ownership_neutral.at[0, 6].set(False),
        )
        episode.match.height = episode.match.width = 8
        episode.frames = [episode.match.frame()]
        with connect(client, 0) as red, connect(client, 1) as blue:
            act(red, receive(red, "observation"), [0, 0, 6, 3, 0])
            act(blue, receive(blue, "observation"))
            result = receive(red, "final")["result"]
            assert result["scores"] == scores and result["reason"] == reason
            assert result["turns"] == 802
            replay = client.get("/replay.json").json()
            assert replay["frames"][-1] == episode.match.frame()


def test_atomic_file_artifacts_and_encoded_paths(tmp_path):
    path = tmp_path / "a folder" / "result.json"
    write_json(path.as_uri(), {"scores": [1, -1]})
    assert read_json(path.as_uri()) == {"scores": [1, -1]}
    assert list(path.parent.iterdir()) == [path]
    with pytest.raises(ValueError):
        write_json("ftp://example.invalid/artifact", {})


def test_artifact_write_failure_cannot_become_success(tmp_path, monkeypatch):
    def broken_write(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("integrations.softmax.server.write_json", broken_write)
    with client_for(tmp_path, max_turns=1) as client, connect(client, 0) as red, connect(client, 1) as blue:
        act(red, receive(red, "observation"))
        act(blue, receive(blue, "observation"))
        receive(red, "failure")
        assert client.app.state.episode.failed
        assert client.get("/replay.json").status_code == 409
        assert not (tmp_path / "results.json").exists()
