"""Browser contract checks for FFA identity and castle construction controls."""

import functools
import json
import os
import re
import shutil
import subprocess
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

if os.environ.get("GENERALS_BROWSER_TESTS") != "1":
    pytest.skip("opt-in browser check; install playwright and Chromium", allow_module_level=True)
pytest.importorskip("playwright")
from playwright.sync_api import expect, sync_playwright

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def viewer(tmp_path):
    bundle = tmp_path / "viewer"
    subprocess.run([str(ROOT / "integrations/softmax/tools/build_replay_viewer.sh"), str(bundle)], check=True)

    class Handler(SimpleHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            if self.path.startswith("/client/player?"):
                self.path = "/index.html"
            elif self.path.startswith("/client/static/"):
                self.path = self.path.removeprefix("/client/static")
            super().do_GET()

    http = ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Handler, directory=bundle))
    threading.Thread(target=http.serve_forever, daemon=True).start()
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(
                executable_path=os.environ.get("CHROMIUM_PATH") or shutil.which("chromium"),
                args=["--no-sandbox"],
            )
            page = browser.new_page(viewport={"width": 1100, "height": 1000})
            page.clock.install(time=0)
            page.clock.pause_at(1)
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            yield page, f"http://127.0.0.1:{http.server_port}", bundle
            assert not errors
            browser.close()
    finally:
        http.shutdown()
        http.server_close()


def connect_player(page, base, players, ruleset="classic", slot=0):
    sockets, actions = [], []

    def connected(ws):
        sockets.append(ws)
        ws.on_message(lambda raw: actions.append(json.loads(raw)))

    page.route_web_socket("**/player?*", connected)
    page.goto(f"{base}/client/player?slot={slot}&token=test")
    expect(page.locator("#mode")).to_have_text("PLAYER")
    sockets[0].send(json.dumps({"type": "hello", "slot": slot, "players": players, "ruleset": ruleset}))
    return sockets[0], actions


def observation(players, slot=0, ruleset="classic"):
    return {
        "type": "observation", "turn": 1, "slot": slot, "players": players,
        "ruleset": ruleset, "eliminated": False, "turn_timeout_seconds": 3600,
        "type_grid": [[1] * 8 for _ in range(6)],
        "owner_grid": [[0] * 8 for _ in range(6)],
        "visible_owner_grid": [[0] * 8 for _ in range(6)],
        "army_grid": [[0] * 8 for _ in range(6)],
        "public_scores": {
            "army": [100 + i for i in range(len(players))],
            "land": [10 + i for i in range(len(players))],
            "eliminated": [False] * len(players),
        },
    }


def test_ffa_visible_owners_scores_and_elimination(viewer):
    page, base, _ = viewer
    players = ["Red", "Blue", "Green", "Purple"]
    ws, actions = connect_player(page, base, players, slot=2)
    obs = observation(players, slot=2)
    for i in range(4):
        obs["owner_grid"][2][i] = 1 if i == 2 else 2
        obs["visible_owner_grid"][2][i] = i + 1
        obs["army_grid"][2][i] = 20
    # Even a malformed owner value cannot color or expose an unrevealed tile.
    obs["type_grid"][0][0] = 0
    obs["visible_owner_grid"][0][0] = 4
    ws.send(json.dumps(obs))
    expect(page.locator("#turn")).to_have_text("1")
    expect(page.locator("#variant-title")).to_have_text("4-PLAYER FFA")
    expect(page.locator(".matchbar .player")).to_have_count(4)
    for i, color in enumerate(["red", "blue", "green", "purple"]):
        expect(page.locator("#board > .tile").nth(16 + i)).to_have_class(re.compile(rf"\b{color}\b"))
        expect(page.locator(f"#army{i}")).to_have_text(str(100 + i))
        expect(page.locator(f"#land{i}")).to_have_text(str(10 + i))
    expect(page.locator("#board > .tile").nth(0)).to_have_class("tile fog")
    page.locator("#board > .tile").nth(18).click()
    page.keyboard.press("ArrowDown")
    assert actions[-1]["action"] == [0, 2, 2, 1, 0]
    page.keyboard.press("ArrowRight")
    expect(page.locator("#queue-count")).to_have_text("1 queued")
    obs["turn"] = 2
    obs["eliminated"] = True
    obs["public_scores"]["eliminated"][2] = True
    ws.send(json.dumps(obs))
    expect(page.locator("#status")).to_contain_text("You have been eliminated")
    expect(page.locator("#name2").locator("../..")).to_have_class(re.compile("eliminated"))
    expect(page.locator("#play-controls")).to_be_hidden()
    expect(page.locator(".tile.selected")).to_have_count(0)
    expect(page.locator(".move-arrow")).to_have_count(0)
    page.keyboard.press("ArrowRight")
    page.clock.run_for(100)
    assert len(actions) == 1
    # FFA cards fit narrow screens without horizontal page scrolling.
    page.set_viewport_size({"width": 390, "height": 800})
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")


def test_castle_queue_cost_and_failure_receipt(viewer):
    page, base, _ = viewer
    players = ["Builder", "Opponent"]
    ws, actions = connect_player(page, base, players, ruleset="build_castles")
    obs = observation(players, ruleset="build_castles")
    obs["build_cost_grid"] = [[0] * 8 for _ in range(6)]
    for r, c in [(2, 1), (4, 4)]:
        obs["owner_grid"][r][c] = obs["visible_owner_grid"][r][c] = 1
        obs["army_grid"][r][c] = 50
        obs["build_cost_grid"][r][c] = 47

    def observe(turn):
        obs["turn"] = turn
        ws.send(json.dumps(obs))
        expect(page.locator("#turn")).to_have_text(str(turn))

    observe(1)
    expect(page.locator("#build")).to_be_visible()
    expect(page.locator("#build")).to_be_disabled()
    expect(page.locator("#variant-title")).to_have_text("BUILD YOUR CASTLES")
    page.locator("#board > .tile").nth(17).click()
    expect(page.locator("#build-hint")).to_have_text("Castle cost: 47 army · 50 on this tile")
    page.keyboard.press("b")
    assert actions == [{"type": "action", "turn": 1, "action": [2, 2, 1, 0, 0]}]
    page.keyboard.press("ArrowRight")
    page.locator("#board > .tile").nth(36).click()
    page.keyboard.press("ArrowRight")
    expect(page.locator("#queue-count")).to_have_text("2 queued")
    # If a queued build loses its army before resolution, only its route stops.
    obs["last_build_executed"] = False
    observe(2)
    assert actions[-1] == {"type": "action", "turn": 2, "action": [0, 4, 4, 3, 0]}
    expect(page.locator("#queue-count")).to_have_text("0 queued")
    page.keyboard.press("Space")
    obs["last_build_executed"] = None
    obs["last_move_executed"] = True
    obs["visible_owner_grid"][4][5] = 1
    obs["army_grid"][4][5] = 49
    obs["army_grid"][2][1] = 46
    observe(3)
    page.locator("#board > .tile").nth(17).click()
    expect(page.locator("#build")).to_be_disabled()
    before = len(actions)
    page.keyboard.press("b")
    expect(page.locator("#status")).to_contain_text("need 47 army")
    assert len(actions) == before
    obs["army_grid"][2][1] = 50
    observe(4)
    page.locator("#build").click()
    assert actions[-1]["action"] == [2, 2, 1, 0, 0]
    obs["type_grid"][2][1] = 3
    obs["army_grid"][2][1] = 3
    obs["build_cost_grid"][2][1] = 0
    obs["last_build_executed"] = True
    observe(5)
    expect(page.locator("#status")).to_have_text("Castle built.")
    expect(page.locator("#board > .tile").nth(17)).to_have_class(re.compile("has-castle"))
    expect(page.locator("#build")).to_be_disabled()


def test_ffa_replay_player_colors_and_elimination(viewer):
    page, base, bundle = viewer
    frame = {
        "turn": 12, "army": [0, 10, 20, 30], "land": [0, 2, 3, 4],
        "eliminated": [True, False, False, False],
        "type_grid": [[4, 4, 4, 4]], "owner_grid": [[1, 2, 3, 4]], "army_grid": [[0, 10, 20, 30]],
    }
    data = {
        "format": "generals-coworld", "version": 1, "ruleset": "classic",
        "height": 1, "width": 4, "players": ["Red", "Blue", "Green", "Purple"],
        "frames": [frame], "result": {"winner": 3, "reason": "general_capture"},
    }
    (bundle / "ffa.json").write_text(json.dumps(data))
    page.goto(f"{base}/#replay=ffa.json")
    expect(page.locator("#mode")).to_have_text("REPLAY")
    expect(page.locator("#status")).to_contain_text("Purple wins")
    expect(page.locator(".matchbar .player.eliminated")).to_have_count(1)
    for i, color in enumerate(["red", "blue", "green", "purple"]):
        expect(page.locator("#board > .tile").nth(i)).to_have_class(re.compile(rf"\b{color}\b"))
    expect(page.locator("#play-controls")).to_be_hidden()
    expect(page.locator("#build")).to_have_count(0)


def test_competition_replay_accepts_2000_turns(viewer):
    page, base, bundle = viewer
    frame = {
        "turn": 0, "army": [1, 1], "land": [1, 1], "eliminated": [False, False],
        "type_grid": [[4, 4]], "owner_grid": [[1, 2]], "army_grid": [[1, 1]],
    }
    data = {
        "format": "generals-coworld", "version": 1, "ruleset": "classic",
        "height": 1, "width": 2, "players": ["Red", "Blue"],
        "frames": [{**frame, "turn": turn} for turn in range(2001)],
        "result": {"winner": -1, "reason": "turn_limit"},
    }
    (bundle / "competition-2000.json").write_text(json.dumps(data))
    page.goto(f"{base}/#replay=competition-2000.json")
    expect(page.locator("#mode")).to_have_text("REPLAY")
    expect(page.locator("#seek")).to_have_attribute("max", "2000")
