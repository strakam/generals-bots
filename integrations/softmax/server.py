"""Authoritative, bounded 1v1 Coworld HTTP/WebSocket server."""

import asyncio
import hmac
import json
import logging
import os
import secrets
from contextlib import asynccontextmanager, suppress
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .artifacts import read_json, write_json
from .config import GameConfig
from .engine import Match, RULESET
from .protocol import PASS, VERSION, parse_action

STATIC = Path(__file__).parent / "static"
LOG = logging.getLogger("generals.softmax")


def enqueue(queue: asyncio.Queue, message: dict):
    # Slow viewers/players must never block the authoritative clock.
    if queue.full():
        queue.get_nowait()
    queue.put_nowait(message)


class Episode:
    def __init__(self, config: GameConfig, match: Match, artifacts: dict, shutdown=None):
        self.config, self.match = config, match
        self.artifacts, self.shutdown = artifacts, shutdown
        self.names = [p.name for p in config.players]
        self.phase = "waiting"
        self.players: dict[int, asyncio.Queue] = {}
        self.reserved_slots: set[int] = set()
        self.viewers: set[asyncio.Queue] = set()
        self.joined = asyncio.Event()
        self.actions_ready = asyncio.Event()
        self.pending: dict[int, list[int]] = {}
        self.timeouts = [0, 0]
        self.consecutive_timeouts = [0, 0]
        self.frames = [match.frame()]
        self.turns = []
        self.result = None
        self.replay = None
        self.failed = False

    def authorize(self, slot: str | None, token: str | None) -> int | None:
        if slot not in ("0", "1") or token is None:
            return None
        index = int(slot)
        return index if hmac.compare_digest(token.encode(), self.config.tokens[index].encode()) else None

    def public(self):
        # This route is reachable by competing players too. Never include live
        # tiles, config, tokens, seed, pending actions, or per-player observations.
        frame = self.frames[-1]
        return {
            "type": "global",
            "protocol_version": VERSION,
            "phase": self.phase,
            "turn": frame["turn"],
            "max_turns": self.config.max_turns,
            "players": self.names,
            "army": frame["army"],
            "land": frame["land"],
            "result": self.result,
            "board": frame if self.phase == "finished" else None,
        }

    def publish(self):
        public = self.public()
        for queue in self.viewers:
            enqueue(queue, public)

    def observation(self, slot):
        return {
            **self.match.observation(slot),
            "players": self.names,
            "max_turns": self.config.max_turns,
            "turn_timeout_seconds": self.config.turn_timeout_seconds,
        }

    def accept(self, slot, message):
        if self.phase != "playing":
            raise ValueError("game is not accepting actions")
        action = parse_action(message, self.match.turn, self.match.height, self.match.width)
        if slot in self.pending:
            raise ValueError("only the first valid action per turn is accepted")
        self.pending[slot] = action
        if len(self.pending) == 2:
            self.actions_ready.set()

    async def save(self, name, payload):
        uri = self.artifacts.get(name)
        if not uri:
            raise RuntimeError(f"missing {name} artifact destination")
        method = self.artifacts.get(f"{name}_method", "PUT")
        await asyncio.to_thread(write_json, uri, payload, method)

    async def run(self):
        try:
            await self.play()
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - process boundary; redact secret-bearing library errors
            # Config and presigned URLs can occur in library exception strings.
            LOG.error("episode failed (%s)", type(exc).__name__)
            self.failed = True
            self.phase = "failed"
            self.publish()
        finally:
            if self.phase in ("finished", "failed"):
                for queue in self.players.values():
                    enqueue(queue, {"type": "final" if not self.failed else "failure", "result": self.result})
                if self.shutdown:
                    # Let final WebSocket messages drain before server teardown.
                    await asyncio.sleep(1)
                    self.shutdown()

    async def play(self):
        try:
            async with asyncio.timeout(self.config.player_connect_timeout_seconds):
                while len(self.players) < 2:
                    self.joined.clear()
                    await self.joined.wait()
        except TimeoutError:
            missing = next(slot for slot in range(2) if slot not in self.players)
            await self.save(
                "failure",
                {"message": "Player did not connect before the start deadline", "failed_policy_index": missing},
            )
            self.failed = True
            self.phase = "failed"
            self.publish()
            return

        winner, reason = -1, "turn_limit"
        while self.match.turn < self.config.max_turns:
            observations = await asyncio.to_thread(lambda: [self.observation(s) for s in range(2)])
            self.pending.clear()
            self.actions_ready.clear()
            self.phase = "playing"
            started = asyncio.get_running_loop().time()
            for slot, queue in self.players.items():
                enqueue(queue, observations[slot])
            self.publish()
            try:
                async with asyncio.timeout(self.config.turn_timeout_seconds):
                    await self.actions_ready.wait()
            except TimeoutError:
                pass
            self.phase = "resolving"
            actions = [self.pending.get(slot, PASS.copy()) for slot in range(2)]
            missing = [slot not in self.pending for slot in range(2)]
            for slot in range(2):
                self.timeouts[slot] += int(missing[slot])
                self.consecutive_timeouts[slot] = self.consecutive_timeouts[slot] + 1 if missing[slot] else 0
            forfeits = [s for s in range(2) if self.consecutive_timeouts[s] >= self.config.max_consecutive_timeouts]
            self.turns.append(
                {"turn": self.match.turn, "actions": actions, "timed_out": missing, "applied": not bool(forfeits)}
            )
            if forfeits:
                winner = 1 - forfeits[0] if len(forfeits) == 1 else -1
                reason = "forfeit" if len(forfeits) == 1 else "double_forfeit"
                break
            winner = await asyncio.to_thread(self.match.advance, actions)
            self.frames.append(await asyncio.to_thread(self.match.frame))
            if winner >= 0:
                reason = "general_capture"
                break
            remaining = self.config.tick_interval_seconds - (asyncio.get_running_loop().time() - started)
            if remaining > 0:
                await asyncio.sleep(remaining)

        self.phase = "saving"
        result = {
            "scores": [0, 0] if winner == -1 else [1 if s == winner else -1 for s in range(2)],
            "winner": winner,
            "reason": reason,
            "turns": self.match.turn,
            "army": self.frames[-1]["army"],
            "land": self.frames[-1]["land"],
            "timeouts": self.timeouts,
        }
        replay = {
            "format": "generals-coworld",
            "version": VERSION,
            "ruleset": RULESET,
            "seed": self.config.seed,
            "height": self.match.height,
            "width": self.match.width,
            "players": self.names,
            "frames": self.frames,
            "turns": self.turns,
            "result": result,
        }
        # Results are the completion marker. Never mark success without a replay.
        await self.save("replay", replay)
        await self.save("results", result)
        self.result, self.replay = result, replay
        self.phase = "finished"
        self.publish()
        LOG.info("episode complete: %s, turn %d, winner %d", reason, self.match.turn, winner)


async def send_messages(ws, queue):
    while True:
        message = await queue.get()
        async with asyncio.timeout(2):
            await ws.send_json(message)
        if message.get("type") in ("final", "failure"):
            await ws.close(code=1000)
            return


async def socket_session(ws, queue, receive):
    sender = asyncio.create_task(send_messages(ws, queue))
    receiver = asyncio.create_task(receive())
    try:
        await asyncio.wait((sender, receiver), return_when=asyncio.FIRST_COMPLETED)
    finally:
        for task in (sender, receiver):
            task.cancel()
        await asyncio.gather(sender, receiver, return_exceptions=True)


def create_app(config: GameConfig | None = None, artifacts: dict | None = None, shutdown=None):
    @asynccontextmanager
    async def lifespan(app):
        try:
            cfg = config or GameConfig.model_validate(
                await asyncio.to_thread(read_json, os.environ["COGAME_CONFIG_URI"])
            )
        except Exception as exc:  # noqa: BLE001 - startup tracebacks must not expose config URLs or tokens
            raise RuntimeError(f"cannot load game config ({type(exc).__name__})") from None
        if cfg.seed is None:
            cfg = cfg.model_copy(update={"seed": secrets.randbits(32)})
        destinations = (
            artifacts
            if artifacts is not None
            else {
                "results": os.environ["COGAME_RESULTS_URI"],
                "replay": os.environ["COGAME_SAVE_REPLAY_URI"],
                "failure": os.environ.get("COGAME_PLAYER_FAILURE_URI"),
                "results_method": os.environ.get("COGAME_RESULTS_METHOD", "PUT"),
                "replay_method": os.environ.get("COGAME_SAVE_REPLAY_METHOD", "PUT"),
                "failure_method": os.environ.get("COGAME_PLAYER_FAILURE_METHOD", "PUT"),
            }
        )
        match = await asyncio.to_thread(Match, cfg.seed)
        episode = Episode(cfg, match, destinations, shutdown)
        app.state.episode = episode
        task = asyncio.create_task(episode.run())
        try:
            yield
        finally:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    # Use the engine's original sprites; the replay build copies the same files.
    app.mount(
        "/static/assets", StaticFiles(directory=Path(__file__).parents[2] / "generals/assets/images"), name="sprites"
    )
    app.mount("/static/fonts", StaticFiles(directory=Path(__file__).parents[2] / "generals/assets/fonts"), name="fonts")
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.middleware("http")
    async def privacy_headers(request, call_next):
        response = await call_next(request)
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    @app.get("/healthz")
    async def health():
        return {"ready": True}

    @app.get("/client/player")
    async def player_client(slot: str = "", token: str = ""):
        if app.state.episode.authorize(slot, token) is None:
            raise HTTPException(403, "invalid player credentials")
        return FileResponse(STATIC / "index.html")

    @app.get("/client/global")
    @app.get("/client/replay")
    async def global_client():
        return FileResponse(STATIC / "index.html")

    @app.get("/replay.json")
    async def replay_data():
        episode = app.state.episode
        if episode.phase != "finished":
            raise HTTPException(409, "replay is available when the episode completes")
        return JSONResponse(episode.replay)

    @app.websocket("/player")
    async def player(ws: WebSocket):
        episode = app.state.episode
        slot = episode.authorize(ws.query_params.get("slot"), ws.query_params.get("token"))
        if slot is None or slot in episode.reserved_slots or episode.phase in ("saving", "finished", "failed"):
            await ws.close(code=1008)
            return
        queue = asyncio.Queue(maxsize=4)
        episode.reserved_slots.add(slot)
        try:
            await ws.accept()
        except Exception:
            episode.reserved_slots.discard(slot)
            raise
        episode.players[slot] = queue
        enqueue(
            queue,
            {
                "type": "hello",
                "protocol_version": VERSION,
                "slot": slot,
                "height": episode.match.height,
                "width": episode.match.width,
                "players": episode.names,
                "ruleset": RULESET,
            },
        )
        if episode.phase == "playing":
            enqueue(queue, episode.observation(slot))
        episode.joined.set()

        async def receive():
            while True:
                raw = await ws.receive_text()
                if len(raw) > 2048:
                    await ws.close(code=1009)
                    return
                try:
                    episode.accept(slot, json.loads(raw))
                except (ValueError, TypeError) as exc:
                    enqueue(
                        queue,
                        {
                            "type": "error",
                            "message": str(exc) if not isinstance(exc, json.JSONDecodeError) else "invalid JSON",
                        },
                    )

        try:
            await socket_session(ws, queue, receive)
        finally:
            if episode.players.get(slot) is queue:
                del episode.players[slot]
            episode.reserved_slots.discard(slot)

    @app.websocket("/global")
    async def global_viewer(ws: WebSocket):
        episode = app.state.episode
        await ws.accept()
        queue = asyncio.Queue(maxsize=1)
        episode.viewers.add(queue)
        enqueue(queue, episode.public())

        async def receive():
            # Spectators cannot pause the engine, alter pacing, or choose seeds.
            while True:
                await ws.receive_text()

        try:
            await socket_session(ws, queue, receive)
        finally:
            episode.viewers.discard(queue)

    return app


def main():
    keep_open = os.environ.get("GENERALS_KEEP_OPEN") == "1"
    app = create_app(shutdown=None if keep_open else lambda: setattr(server, "should_exit", True))
    server = uvicorn.Server(
        uvicorn.Config(
            app,
            host=os.environ.get("COGAME_HOST", "0.0.0.0"),
            port=int(os.environ.get("COGAME_PORT", "8080")),
            # Uvicorn's websocket INFO messages contain query strings even when
            # access_log=False. Suppress them so slot tokens never reach game logs.
            access_log=False,
            log_level="warning",
            ws="websockets",
            ws_max_size=2048,
            ws_max_queue=8,
        )
    )
    server.run()
    raise SystemExit(1 if not hasattr(app.state, "episode") or app.state.episode.failed else 0)


if __name__ == "__main__":
    main()
