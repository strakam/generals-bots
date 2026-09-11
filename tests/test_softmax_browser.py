"""Opt-in real browser checks: GENERALS_BROWSER_TESTS=1 pytest -q tests/test_softmax_browser.py."""

import asyncio
import functools
import gzip
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

if os.environ.get("GENERALS_BROWSER_TESTS") != "1":
    pytest.skip("opt-in browser check; install playwright and Chromium", allow_module_level=True)
pytest.importorskip("playwright")
import httpx
from playwright.sync_api import expect, sync_playwright
from websockets.asyncio.client import connect

ROOT = Path(__file__).resolve().parents[1]


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


async def ping_global(port):
    async with connect(f"ws://127.0.0.1:{port}/global") as ws:
        message = json.loads(await ws.recv())
        assert message["board"] is None
        pong = await ws.ping(b"coworld-certification-probe")
        await asyncio.wait_for(pong, timeout=2)


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


def test_browser_premove_controls(tmp_path):
    """Drive turns explicitly to check fast inputs between observations deterministically."""
    bundle = tmp_path / "viewer"
    subprocess.run([str(ROOT / "integrations/softmax/tools/build_replay_viewer.sh"), str(bundle)], check=True)

    class PlayerHandler(QuietHandler):
        def do_GET(self):
            if self.path.startswith("/client/player?"):
                self.path = "/index.html"
            elif self.path.startswith("/static/"):
                self.path = self.path.removeprefix("/static")
            super().do_GET()

    http = ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(PlayerHandler, directory=bundle))
    threading.Thread(target=http.serve_forever, daemon=True).start()
    try:
        with sync_playwright() as p:
            executable = os.environ.get("CHROMIUM_PATH") or shutil.which("chromium")
            browser = p.chromium.launch(executable_path=executable, args=["--no-sandbox"])
            page = browser.new_page(viewport={"width": 1100, "height": 1000})
            page.clock.install()
            sockets, actions, errors = [], [], []
            page.on("pageerror", lambda error: errors.append(str(error)))

            def connected(ws):
                sockets.append(ws)
                ws.on_message(lambda raw: actions.append(json.loads(raw)))

            page.route_web_socket("**/player?*", connected)
            page.goto(f"http://127.0.0.1:{http.server_port}/client/player?slot=0&token=test")
            expect(page.locator("#mode")).to_have_text("PLAYER")
            sockets[0].send(json.dumps({"type": "hello", "slot": 0, "players": ["Human", "Bot"]}))
            obs = {
                "type": "observation", "slot": 0, "players": ["Human", "Bot"],
                "height": 6, "width": 8, "my_army": 10, "opp_army": 10,
                "my_land": 1, "opp_land": 1, "turn_timeout_seconds": 3600,
                "type_grid": [[1] * 8 for _ in range(6)],
                "owner_grid": [[0] * 8 for _ in range(6)],
                "army_grid": [[0] * 8 for _ in range(6)],
            }
            obs["type_grid"][2][1] = 4
            obs["owner_grid"][2][1] = 1
            obs["army_grid"][2][1] = 10

            def observe(turn, r=2, c=1, army=10):
                obs["turn"] = turn
                obs["owner_grid"][r][c] = 1
                obs["army_grid"][r][c] = army
                sockets[0].send(json.dumps(obs))
                expect(page.locator("#turn")).to_have_text(str(turn))

            def selected(r, c):
                expect(page.locator("#board > .tile").nth(r * 8 + c)).to_have_class(re.compile(r"\bselected\b"))

            observe(1)
            for key in ["ArrowRight", "ArrowRight", "ArrowDown", "ArrowLeft"]:
                page.keyboard.press(key)
            expect(page.locator("#queue-count")).to_have_text("3 queued")
            expect(page.locator(".move-arrow.queued")).to_have_count(3)
            expect(page.locator(".move-arrow.submitted")).to_have_count(1)
            selected(3, 2)
            assert actions == [{"type": "action", "turn": 1, "action": [0, 2, 1, 3, 0]}]
            page.keyboard.press("e")
            expect(page.locator("#queue-count")).to_have_text("2 queued")
            expect(page.locator(".move-arrow.queued")).to_have_count(2)
            selected(3, 3)
            page.keyboard.press("h")
            page.keyboard.press("ArrowLeft")
            page.screenshot(path=str(tmp_path / "premoves.png"), full_page=True)
            page.keyboard.press("q")
            expect(page.locator("#queue-count")).to_have_text("0 queued")
            expect(page.locator(".move-arrow")).to_have_count(1)
            selected(2, 2)
            assert len(actions) == 1  # Q cannot retract this turn's submitted action.

            for key in ["ArrowRight", "ArrowDown", "ArrowLeft"]:
                page.keyboard.press(key)
            page.keyboard.press("h")  # Already queued moves keep their half-army setting.
            observe(2, 2, 2, 8)
            expect(page.locator("#queue-count")).to_have_text("2 queued")
            selected(3, 2)  # The planned endpoint is still unowned.
            observe(3, 2, 3, 4)
            expect(page.locator("#queue-count")).to_have_text("1 queued")
            observe(4, 3, 3, 2)
            expect(page.locator("#queue-count")).to_have_text("0 queued")
            assert [a["action"] for a in actions] == [
                [0, 2, 1, 3, 0], [0, 2, 2, 3, 1], [0, 2, 3, 1, 1], [0, 3, 3, 2, 1],
            ]
            assert [a["turn"] for a in actions] == [1, 2, 3, 4]

            # A premove waits for reinforcements instead of losing the route.
            obs["turn_timeout_seconds"] = 1
            observe(5, 3, 2, 1)
            page.keyboard.press("ArrowDown")
            expect(page.locator("#queue-count")).to_have_text("1 queued")
            assert len(actions) == 4
            page.clock.run_for(800)
            assert actions[-1] == {"type": "action", "turn": 5, "action": [1, 0, 0, 0, 0]}
            expect(page.locator("#queue-count")).to_have_text("1 queued")
            obs["turn_timeout_seconds"] = 3600
            observe(6, 3, 2, 3)
            expect(page.locator("#queue-count")).to_have_text("0 queued")
            assert actions[-1]["action"] == [0, 3, 2, 1, 0]

            # A failed capture stops the rest of the path; no moves from unowned tiles.
            page.keyboard.press("ArrowRight")
            observe(7)
            expect(page.locator("#queue-count")).to_have_text("0 queued")
            expect(page.locator("#status")).to_contain_text("no longer yours")
            expect(page.locator(".move-arrow")).to_have_count(0)
            assert len(actions) == 6

            # Mouse input, keyboard focus after buttons, and visible mountains.
            page.locator("#board > .tile").nth(2 * 8 + 1).click()
            page.locator("#board > .tile").nth(2 * 8 + 2).click()
            page.keyboard.press("ArrowRight")
            page.click("#undo")
            page.keyboard.press("ArrowDown")
            expect(page.locator("#queue-count")).to_have_text("1 queued")
            page.click("#clear")
            expect(page.locator("#queue-count")).to_have_text("0 queued")
            obs["type_grid"][1][2] = 2
            observe(8, 2, 2, 8)
            page.keyboard.press("ArrowUp")
            expect(page.locator(".move-arrow")).to_have_count(0)
            selected(2, 2)

            # The mountain-shaped obstacles in fog must reject keyboard AND mouse input.
            obs["type_grid"][1][2] = 5
            observe(9, 2, 2, 8)
            before = len(actions)
            page.keyboard.press("ArrowUp")
            page.locator("#board > .tile").nth(1 * 8 + 2).click()
            expect(page.locator("#queue-count")).to_have_text("0 queued")
            expect(page.locator(".move-arrow")).to_have_count(0)
            selected(2, 2)
            assert len(actions) == before

            # A newly revealed obstacle trims only the blocked suffix of a route.
            for key in ["ArrowRight", "ArrowDown", "ArrowRight", "ArrowUp"]:
                page.keyboard.press(key)
            expect(page.locator("#queue-count")).to_have_text("3 queued")
            obs["type_grid"][2][4] = 5
            observe(10, 2, 3, 8)
            expect(page.locator("#queue-count")).to_have_text("1 queued")
            expect(page.locator(".move-arrow[data-direction='0']")).to_have_count(0)
            selected(3, 4)
            assert actions[-1]["action"] == [0, 2, 3, 1, 0]
            page.keyboard.press("q")

            # Once revealed as a castle, the fog obstacle is a valid attack target.
            obs["type_grid"][1][2] = 3
            observe(11, 3, 3, 7)
            page.locator("#board > .tile").nth(2 * 8 + 2).click()
            page.keyboard.press("ArrowUp")
            assert actions[-1]["action"] == [0, 2, 2, 0, 0]
            observe(12, 1, 2, 6)

            # Disconnect discards local plans and prevents further submissions.
            page.keyboard.press("ArrowRight")
            page.keyboard.press("ArrowDown")
            expect(page.locator("#queue-count")).to_have_text("1 queued")
            sockets[0].close()
            expect(page.locator("#status")).to_contain_text("Connection closed")
            expect(page.locator("#queue-count")).to_have_text("0 queued")
            expect(page.locator(".move-arrow")).to_have_count(0)
            page.keyboard.press("ArrowRight")
            expect(page.locator("#queue-count")).to_have_text("0 queued")
            assert not errors

            # Visual review of every direction over ground, both owners, fog and castles.
            page.evaluate("""() => {
                const frame = {
                    type_grid: [1, 1, 1, 0, 3].map(kind => Array(4).fill(kind)),
                    owner_grid: [0, 1, 2, 0, 0].map(owner => Array(4).fill(owner)),
                    army_grid: Array.from({length: 5}, () => Array(4).fill(123)),
                };
                const arrows = Array.from({length: 5}, (_, r) =>
                    Array.from({length: 4}, (_, c) => [0, r, c, c, 0])).flat();
                GeneralsTiles(document.getElementById('board')).draw(frame, null, arrows);
            }""")
            page.locator("#board").screenshot(path=str(tmp_path / "arrow-contrast.png"))
            browser.close()
    finally:
        http.shutdown()
        http.server_close()


def test_browser_play_and_standalone_replay(tmp_path):
    port = free_port()
    config = {
        "tokens": ["browser-red", "browser-blue"],
        "players": [{"name": "Human"}, {"name": "Expander"}],
        "seed": 7,
        "max_turns": 16,
        "tick_interval_seconds": 0.5,
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    env = {
        **os.environ,
        "JAX_PLATFORMS": "cpu",
        "COGAME_CONFIG_URI": (tmp_path / "config.json").as_uri(),
        "COGAME_RESULTS_URI": (tmp_path / "results.json").as_uri(),
        "COGAME_SAVE_REPLAY_URI": (tmp_path / "replay.json").as_uri(),
        "COGAME_PLAYER_FAILURE_URI": (tmp_path / "failure.json").as_uri(),
        "COGAME_HOST": "127.0.0.1",
        "COGAME_PORT": str(port),
        "GENERALS_KEEP_OPEN": "1",
    }
    processes = []
    try:
        with (tmp_path / "game.log").open("w") as log:
            server = subprocess.Popen(
                [sys.executable, "-m", "integrations.softmax.server"], cwd=ROOT, env=env, stdout=log, stderr=log
            )
        processes.append(server)
        with httpx.Client(timeout=1) as client:
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                if server.poll() is not None:
                    pytest.fail((tmp_path / "game.log").read_text())
                try:
                    if client.get(f"http://127.0.0.1:{port}/healthz").status_code == 200:
                        break
                except httpx.TransportError:
                    pass
                time.sleep(0.1)
            else:
                pytest.fail("game startup timed out")
        asyncio.run(ping_global(port))
        with sync_playwright() as p:
            executable = os.environ.get("CHROMIUM_PATH") or shutil.which("chromium")
            browser = p.chromium.launch(executable_path=executable, headless=True, args=["--no-sandbox"])
            page = browser.new_page(viewport={"width": 1100, "height": 1100})
            errors, observations = [], []
            page.on("pageerror", lambda error: errors.append(str(error)))

            def on_socket(ws):
                def received(raw):
                    message = json.loads(raw)
                    if message.get("type") == "observation":
                        observations.append(message)

                ws.on("framereceived", received)

            page.on("websocket", on_socket)
            page.goto(f"http://127.0.0.1:{port}/client/player?slot=0&token=browser-red")
            player_env = {
                **os.environ,
                "COWORLD_PLAYER_WS_URL": f"ws://127.0.0.1:{port}/player?slot=1&token=browser-blue",
            }
            bot = subprocess.Popen([sys.executable, "-m", "integrations.softmax.player"], cwd=ROOT, env=player_env)
            processes.append(bot)
            page.wait_for_function("Number(document.getElementById('turn').textContent) >= 2")
            obs = observations[-1]
            general = next(
                (r, c)
                for r, row in enumerate(obs["type_grid"])
                for c, kind in enumerate(row)
                if kind == 4 and obs["owner_grid"][r][c] == 1
            )
            r, c = general
            direction = next(
                key
                for dr, dc, key in [(-1, 0, "ArrowUp"), (1, 0, "ArrowDown"), (0, -1, "ArrowLeft"), (0, 1, "ArrowRight")]
                if 0 <= r + dr < obs["height"] and 0 <= c + dc < obs["width"] and obs["type_grid"][r + dr][c + dc] == 1
            )
            page.keyboard.press(direction)
            page.screenshot(path=str(tmp_path / "player.png"), full_page=True)
            page.wait_for_function("document.getElementById('mode').textContent === 'REPLAY'", timeout=20000)
            assert bot.wait(timeout=5) == 0
            recorded = json.loads((tmp_path / "replay.json").read_text())
            assert recorded["result"]["timeouts"] == [0, 0]
            assert any(turn["actions"][0][0] == 0 for turn in recorded["turns"])
            assert not errors
            page.close()

            # No game backend remains: replay rendering must be entirely static.
            server.terminate()
            server.wait(timeout=5)
            bundle = tmp_path / "viewer"
            hook = ROOT / "integrations/softmax/tools/build_replay_viewer.sh"
            subprocess.run([str(hook), str(bundle)], check=True)
            (bundle / "obsolete.txt").write_text("old generated output")
            subprocess.run([str(hook), str(bundle)], check=True)
            assert not (bundle / "obsolete.txt").exists()
            shutil.copyfile(tmp_path / "replay.json", bundle / "replay.json")
            (bundle / "replay.gz").write_bytes(gzip.compress((bundle / "replay.json").read_bytes()))
            (bundle / "broken.json").write_text('{"format":"wrong"}')
            http = ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(QuietHandler, directory=bundle))
            thread = threading.Thread(target=http.serve_forever, daemon=True)
            thread.start()
            try:
                base = f"http://127.0.0.1:{http.server_port}"
                page = browser.new_page(viewport={"width": 1100, "height": 1100})
                page.on("pageerror", lambda error: errors.append(str(error)))
                page.goto(base + "/#replay=replay.json")
                page.wait_for_function("Number(document.getElementById('seek').value) > 0")
                page.click("#pause")
                page.locator("#seek").fill("8")
                assert page.locator("#turn").inner_text() == "8"
                page.screenshot(path=str(tmp_path / "replay-desktop.png"), full_page=True)
                page.click("#restart")
                assert page.locator("#turn").inner_text() == "0"
                page.set_viewport_size({"width": 390, "height": 844})
                page.screenshot(path=str(tmp_path / "replay-mobile.png"), full_page=True)
                assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
                page.goto(base + "/#replay=replay.gz")
                page.wait_for_function("document.getElementById('mode').textContent === 'REPLAY'")
                page.goto(base + "/#replay=broken.json")
                page.wait_for_function("document.getElementById('status').classList.contains('error')")
                assert "Unsupported" in page.locator("#status").inner_text()
                assert not errors
            finally:
                http.shutdown()
                http.server_close()
                browser.close()
        assert "browser-red" not in (tmp_path / "game.log").read_text()
        assert "browser-blue" not in (tmp_path / "game.log").read_text()
    finally:
        for process in reversed(processes):
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
