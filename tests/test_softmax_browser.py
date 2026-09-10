"""Opt-in real browser checks: GENERALS_BROWSER_TESTS=1 pytest -q tests/test_softmax_browser.py."""

import asyncio
import functools
import gzip
import json
import os
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
from playwright.sync_api import sync_playwright
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
