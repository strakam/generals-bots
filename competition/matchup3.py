"""
Human-vs-bot match server: a generals.io-style web client on top of the
competition ruleset.

Runs the same engine path as matchup2.py (build-castles + deathtouch applied,
per-seed rectangular boards) and drives the bot over the ordinary stdio
protocol, so **competition agents run unmodified** — matchup3 is just another
consumer of `competition/protocol.py`.

    python competition/matchup3.py --host 0.0.0.0 --port 8080

Then open http://<host>:8080/ and pick a bot.

Design notes
------------
* Stdlib only (http.server + JSON polling). No FastAPI/Flask/websockets, so a
  VPS needs nothing beyond this repo's existing dependencies. The browser polls
  ~10x/second and the server answers 204 when nothing changed.
* The move queue lives **server-side**, like generals.io: the client appends
  moves and the game loop pops one per tick. A laggy or closed browser just
  means the queue drains and the human passes — the match stays consistent.
* The human only ever receives `game.get_observation(state, seat)`, the same
  fog-limited, perspective-relative view the bot gets. No peeking.
"""
import argparse
import base64
import gzip
import json
import sys
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "competition"))

import numpy as np

from generals import GeneralsEnv
from generals.core import game
from generals.modifiers import build_castles as _build_castles
from matchup2 import make_init_state, make_stepper
from protocol import decode_action, encode_handshake, encode_observation

WEB_DIR = Path(__file__).resolve().parent / "web"
AGENTS_DIR = REPO_ROOT / "competition" / "agents"
# The browser draws with the same artwork as the pygame GUI, so the web board
# looks like the desktop one instead of an approximation.
ASSETS_DIR = REPO_ROOT / "generals" / "assets" / "images"
ASSETS = {"crownie.png", "citie.png", "mountainie.png"}
PASS_ACTION = (1, 0, 0, 0, 0)
BUILD = 2


class Archive:
    """Finished matches kept for replay, newest first, capped at `limit`.

    Only the compressed blob is retained (~100 KB per 1000-turn match), so a
    default of 10 costs about a megabyte.
    """

    def __init__(self, limit=10):
        self.limit = max(1, limit)
        self.items = []          # [{id, meta, blob}], newest first
        self.lock = threading.RLock()
        self._next = 1

    def add(self, meta, blob, rpl=None):
        with self.lock:
            if not blob:
                return None
            rid = f"m{self._next:04d}"
            self._next += 1
            self.items.insert(0, {"id": rid, "meta": meta, "blob": blob,
                                  "rpl": rpl})
            del self.items[self.limit:]
            return rid

    def listing(self):
        with self.lock:
            return [{"id": it["id"], **it["meta"],
                     "kb": len(it["blob"]) // 1024} for it in self.items]

    def get(self, rid=None, key="blob"):
        with self.lock:
            if not self.items:
                return None
            if rid is None:
                return self.items[0][key]
            for it in self.items:
                if it["id"] == rid:
                    return it[key]
            return None


def list_bots():
    return sorted(p.parent.name for p in AGENTS_DIR.glob("*/run.sh"))


class Match:
    """One human-vs-bot game, ticking on its own thread."""

    def __init__(self, bot_name, seed, seat, tick_ms, mode="competition"):
        self.lock = threading.RLock()
        self.bot_name = bot_name
        self.seed = seed
        self.seat = seat                  # human's player index (0 or 1)
        self.bot_seat = 1 - seat
        self.tick_ms = tick_ms
        self.queue = deque()
        self.version = 0
        self.result = None
        self.error = None
        self.turn = 0
        self.last_build_error = None
        self._stop = threading.Event()
        # Full-information frame per turn, kept for replay. Stored as compact
        # numpy arrays (~2.9 KB/turn, so ~3.5 MB for a full 1200-turn game) and
        # only ever released once the match is over — see replay_payload.
        self.history = []
        self._blob = None
        # The action stream, kept separately so a match can also be exported as
        # a real .rpl that competition/replay.py (and the pygame GUI) can open.
        # 10 ints per turn — a rounding error next to the frame history.
        self.actions_log = []
        self._rpl = None
        # Optional bot playing the human's seat (see set_autopilot).
        self.auto_proc = None
        self.auto_name = None

        self.env = GeneralsEnv(mode=mode)
        self.step = make_stepper(self.env)
        self.state, self.dims = make_init_state(self.env, _key(seed))
        self.pad = self.env.pad_to

        run_sh = AGENTS_DIR / bot_name / "run.sh"
        if not run_sh.exists():
            raise FileNotFoundError(f"no agent {bot_name!r}")
        import subprocess
        self.proc = subprocess.Popen(
            ["bash", str(run_sh)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, bufsize=1, text=True, cwd=str(run_sh.parent))
        self.proc.stdin.write(encode_handshake(self.bot_seat, self.pad, self.pad))
        self.proc.stdin.flush()

        self._record()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    # ---------------- game loop ----------------

    def _ask_bot(self, obs):
        self.proc.stdin.write(encode_observation(obs))
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError(f"bot {self.bot_name} closed stdout")
        return decode_action(line)

    def _loop(self):
        period = self.tick_ms / 1000.0
        next_tick = time.monotonic()
        while not self._stop.is_set():
            now = time.monotonic()
            if now < next_tick:
                time.sleep(min(0.01, next_tick - now))
                continue
            next_tick += period
            if next_tick < now:           # fell behind; don't spiral
                next_tick = now + period
            try:
                self._tick()
            except Exception as exc:      # a dead bot must not wedge the server
                with self.lock:
                    self.error = str(exc)
                    self.result = f"aborted: {exc}"
                    self.version += 1
                return

    def _record(self):
        """Snapshot the true board (no fog) for later replay."""
        s = self.state
        H, W = np.asarray(s.armies).shape
        types = np.ones((H, W), dtype=np.int8)
        types[np.asarray(s.mountains, bool)] = 2
        types[np.asarray(s.castles, bool)] = 3
        types[np.asarray(s.generals, bool)] = 4
        owner = np.zeros((H, W), dtype=np.int8)
        owner[np.asarray(s.ownership[self.seat], bool)] = 1      # always "you"
        owner[np.asarray(s.ownership[self.bot_seat], bool)] = 2  # always the bot
        self.history.append((types, owner, np.asarray(s.armies, dtype=np.int32)))

    def set_autopilot(self, bot_name):
        """Hand the human seat to a bot, or take it back (bot_name=None).

        A second agent process is spawned on the player's seat and fed the same
        fog-limited frames the player sees, so the bot plays the position it is
        actually given. Toggling off kills it; the queue takes over again.
        """
        with self.lock:
            if self.auto_proc is not None:
                try:
                    self.auto_proc.stdin.close()
                except Exception:
                    pass
                try:
                    self.auto_proc.wait(timeout=1)
                except Exception:
                    self.auto_proc.kill()
                self.auto_proc = None
                self.auto_name = None
            if not bot_name:
                return True, "autopilot off"

            run_sh = AGENTS_DIR / bot_name / "run.sh"
            if not run_sh.exists():
                return False, f"no agent {bot_name!r}"
            import subprocess
            self.auto_proc = subprocess.Popen(
                ["bash", str(run_sh)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, bufsize=1, text=True, cwd=str(run_sh.parent))
            self.auto_proc.stdin.write(encode_handshake(self.seat, self.pad, self.pad))
            self.auto_proc.stdin.flush()
            self.auto_name = bot_name
            # A bot taking over mid-game inherits none of the queued plan.
            self.queue.clear()
            return True, f"autopilot: {bot_name}"

    def replay_meta(self):
        return {"bot": self.bot_name, "seed": self.seed, "seat": self.seat,
                "turns": self.turn, "result": self.result}

    def rpl_bytes(self):
        """The match as a competition/replay.py recording (.rpl).

        Same format matchup2 --save writes, so an exported file opens in the
        pygame replay GUI. Actions only: replay.py re-simulates from the seed.
        """
        with self.lock:
            if self._rpl is not None:
                return self._rpl
            if self.result is None or not self.actions_log:
                return None
            import replay as replay_io
            meta = {
                "seed": self.seed,
                "mode": "competition",
                "grid_size": None,
                "truncation": int(self.env.truncation),
                "perfect_info": bool(self.env.perfect_info),
                "board": f"{self.dims[0]}x{self.dims[1]}",
                # Seat order matters: index 0 must be player 0.
                "agents": (["you", self.bot_name] if self.seat == 0
                           else [self.bot_name, "you"]),
                "result": self.result,
            }
            self._rpl = replay_io.dumps(meta, self.actions_log)
            self.actions_log = []
            return self._rpl

    def replay_payload(self):
        """gzip+base64 JSON of the whole match, for the replay viewer.

        Fog is applied in the browser rather than here, so one recording can be
        re-rendered from either seat or with no fog at all. Released only after
        the match ends — handing the true board to a live player would be
        handing them the answer.
        """
        with self.lock:
            if self._blob is not None:
                return self._blob
            if self.result is None or not self.history:
                return None
            H, W = self.history[0][0].shape
            blob = {
                "H": H, "W": W, "seat": self.seat, "bot": self.bot_name,
                "result": self.result,
                "frames": [{"t": t.ravel().tolist(),
                            "o": o.ravel().tolist(),
                            "a": a.ravel().tolist()}
                           for t, o, a in self.history],
            }
            raw = json.dumps(blob, separators=(",", ":")).encode()
            self._blob = base64.b64encode(gzip.compress(raw, 6)).decode()
            # The compressed blob is ~30x smaller; keeping both would mean
            # holding several MB per archived match for no reason.
            self.history = []
            return self._blob

    def _tick(self):
        with self.lock:
            if self.result is not None:
                return
            bot_obs = game.get_observation(self.state, self.bot_seat)
            bot_action = self._ask_bot(bot_obs)

            if self.auto_proc is not None:
                # Autopilot sees exactly what the player sees — same seat, same fog.
                human_obs = game.get_observation(self.state, self.seat)
                self.auto_proc.stdin.write(encode_observation(human_obs))
                self.auto_proc.stdin.flush()
                line = self.auto_proc.stdout.readline()
                if not line:
                    raise RuntimeError(f"autopilot {self.auto_name} closed stdout")
                human_action = tuple(int(x) for x in line.split())
            else:
                human_action = self.queue.popleft() if self.queue else PASS_ACTION

            acts = [None, None]
            acts[self.seat] = np.array(human_action, dtype=np.int32)
            acts[self.bot_seat] = np.asarray(bot_action, dtype=np.int32)
            stacked = np.stack(acts)
            self.actions_log.append(stacked.copy())
            self.state, info = self.step(self.state, stacked)
            self.turn += 1
            self.version += 1
            self._record()

            if bool(info.is_done):
                w = int(info.winner)
                if w < 0:
                    self.result = "draw (mutual deathtouch)"
                else:
                    self.result = "you win!" if w == self.seat else f"{self.bot_name} wins"
            elif self.turn >= self.env.truncation:
                self.result = f"draw (truncated at {self.env.truncation})"

    def close(self):
        self._stop.set()
        if self.auto_proc is not None:
            try:
                self.auto_proc.stdin.close()
                self.auto_proc.wait(timeout=1)
            except Exception:
                self.auto_proc.kill()
            self.auto_proc = None
        try:
            self.proc.stdin.close()
        except Exception:
            pass
        try:
            self.proc.wait(timeout=2)
        except Exception:
            self.proc.kill()

    # ---------------- client API ----------------

    def snapshot(self):
        """The human's fog-limited view plus everything the UI needs."""
        with self.lock:
            obs = game.get_observation(self.state, self.seat)
            armies = np.asarray(obs.armies, dtype=np.int32)
            H, W = armies.shape

            type_grid = np.ones((H, W), dtype=np.int32)
            type_grid[np.asarray(obs.fog_cells, bool)] = 0
            type_grid[np.asarray(obs.structures_in_fog, bool)] = 5
            type_grid[np.asarray(obs.mountains, bool)] = 2
            type_grid[np.asarray(obs.castles, bool)] = 3
            type_grid[np.asarray(obs.generals, bool)] = 4

            owner_grid = np.zeros((H, W), dtype=np.int32)
            owner_grid[np.asarray(obs.owned_cells, bool)] = 1
            owner_grid[np.asarray(obs.opponent_cells, bool)] = 2

            cost = np.asarray(
                _build_castles.build_cost_grid(self.state, self.seat), dtype=np.int32)

            return {
                "version": self.version,
                "turn": int(obs.timestep),
                "H": H, "W": W,
                "board": f"{self.dims[0]}x{self.dims[1]}",
                "type": type_grid.tolist(),
                "owner": owner_grid.tolist(),
                "army": armies.tolist(),
                "build_cost": cost.tolist(),
                "my_land": int(obs.owned_land_count),
                "my_army": int(obs.owned_army_count),
                "opp_land": int(obs.opponent_land_count),
                "opp_army": int(obs.opponent_army_count),
                "queued": len(self.queue),
                "result": self.result,
                "bot": self.bot_name,
                "seat": self.seat,
                "tick_ms": self.tick_ms,
                "truncation": int(self.env.truncation),
                "deathtouch_turn": self.env.deathtouch_turn,
                "build_msg": self.last_build_error,
                "autopilot": self.auto_name,
            }

    def enqueue(self, moves):
        with self.lock:
            for r, c, d, split in moves:
                self.queue.append((0, int(r), int(c), int(d), int(split)))
            return len(self.queue)

    def clear_queue(self):
        with self.lock:
            self.queue.clear()

    def request_build(self, r, c):
        """Validate against the engine's own rules before spending a turn.

        The engine silently consumes an invalid build as a pass; for a human
        that is just a lost turn with no explanation, so check first and say
        why. Build jumps the queue — it is a deliberate, immediate action.
        """
        with self.lock:
            if self.result is not None:
                return False, "game over"
            H, W = np.asarray(self.state.armies).shape
            if not (0 <= r < H and 0 <= c < W):
                return False, "off board"
            if not bool(np.asarray(self.state.ownership[self.seat])[r, c]):
                self.last_build_error = "you don't own that cell"
                return False, "you don't own that cell"
            if bool(np.asarray(self.state.generals)[r, c]) or \
                    bool(np.asarray(self.state.castles)[r, c]):
                return False, "already a structure"
            cost = int(np.asarray(
                _build_castles.build_cost_grid(self.state, self.seat))[r, c])
            have = int(np.asarray(self.state.armies)[r, c])
            if have < cost:
                self.last_build_error = f"needs {cost} army, cell has {have}"
                return False, f"needs {cost} army, cell has {have}"
            self.queue.appendleft((BUILD, int(r), int(c), 0, 0))
            self.last_build_error = f"building for {cost}"
            return True, f"building for {cost}"


def _key(seed):
    import jax.random as jrandom
    return jrandom.PRNGKey(seed)


# --------------------------------------------------------------------------
# Worker mode: engine + bot stay here, the public site is somewhere else.
# --------------------------------------------------------------------------

def run_worker(relay_url, token, poll_ms=120):
    """Dial out to a relay and serve browsers through it.

    The laptop is behind NAT, so the connection has to be outbound: we push the
    current snapshot and receive queued browser commands in the same response.
    While no match is running we let the relay hold the request open (it blocks
    up to ~20s), so idle traffic is negligible but Start is still instant.
    """
    import urllib.error
    import urllib.request

    url = relay_url.rstrip("/") + "/agent/sync"
    match = None
    bots = list_bots()
    sent_replay = False
    # Replays for matches abandoned mid-game (a new match started before the
    # old one finished) still deserve to reach the relay.
    pending_replays = []
    print(f"[worker] relaying to {url}")
    print(f"[worker] offering: {', '.join(bots)}")
    warned = False

    while True:
        snapshot = match.snapshot() if match is not None else None
        # A finished match still shows its final board, but there is nothing
        # left to tick — so long-poll instead of spinning at the tick rate.
        idle = match is None or match.result is not None
        payload = {"token": token, "bots": bots, "snapshot": snapshot, "idle": idle}
        if pending_replays and "replay" not in payload:
            meta, blob, rpl = pending_replays.pop(0)
            payload["replay"], payload["replay_meta"] = blob, meta
            if rpl:
                payload["rpl_b64"] = base64.b64encode(rpl).decode()
        # Ship the replay exactly once, after the result is known. It is a few
        # hundred KB, so it must never ride along with the per-tick snapshot.
        if match is not None and match.result is not None and not sent_replay:
            blob = match.replay_payload()
            if blob:
                payload["replay"] = blob
                payload["replay_meta"] = match.replay_meta()
                rpl = match.rpl_bytes()
                if rpl:
                    payload["rpl_b64"] = base64.b64encode(rpl).decode()
                sent_replay = True
                print(f"[worker] replay uploaded ({len(blob) // 1024} KB): "
                      f"{match.replay_meta()['result']}")
        try:
            req = urllib.request.Request(
                url, data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=40) as resp:
                reply = json.loads(resp.read() or b"{}")
            if warned:
                print("[worker] reconnected")
                warned = False
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
            if not warned:
                print(f"[worker] relay unreachable ({exc}); retrying")
                warned = True
            time.sleep(2.0)
            continue

        for cmd in reply.get("commands", []):
            op, args = cmd.get("op"), cmd.get("args", {})
            try:
                if op == "new":
                    if match is not None:
                        if match.result is None:
                            match.result = f"abandoned at turn {match.turn}"
                        blob = match.replay_payload()
                        if blob and not sent_replay:
                            pending_replays.append(
                                (match.replay_meta(), blob, match.rpl_bytes()))
                        match.close()
                    match = Match(bot_name=args.get("bot", bots[0]),
                                  seed=int(args.get("seed", 0)),
                                  seat=int(args.get("seat", 0)),
                                  tick_ms=int(args.get("tick_ms", 250)))
                    sent_replay = False
                    print(f"[worker] new match vs {match.bot_name} "
                          f"seed={match.seed} seat={match.seat}")
                elif match is None:
                    continue
                elif op == "queue":
                    match.enqueue(args.get("moves", []))
                elif op == "clear":
                    match.clear_queue()
                elif op == "build":
                    match.request_build(int(args["r"]), int(args["c"]))
                elif op == "autopilot":
                    ok, msg = match.set_autopilot(args.get("bot") or None)
                    print(f"[worker] {msg}")
                elif op == "resign":
                    if match.result is None:
                        match.result = f"abandoned at turn {match.turn}"
                    blob = match.replay_payload()
                    if blob and not sent_replay:
                        pending_replays.append(
                            (match.replay_meta(), blob, match.rpl_bytes()))
                    match.close()
                    match = None
                    print("[worker] match ended by client")
            except Exception as exc:
                print(f"[worker] command {op} failed: {exc}")

        if not idle:
            time.sleep(poll_ms / 1000.0)


class Server(ThreadingHTTPServer):
    daemon_threads = True
    match: Match | None = None
    match_lock = threading.Lock()
    archive: "Archive" = None


class Handler(BaseHTTPRequestHandler):
    server_version = "matchup3"

    def log_message(self, *args):
        pass                              # the tick loop is the interesting output

    # -------- helpers --------

    def _send(self, code, body=b"", ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        origin = getattr(self.server, "cors", None)
        if origin:
            # Needed only when the page is served from somewhere else
            # (e.g. Vercel) and calls this box as an API.
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()
        if body:
            self.wfile.write(body)

    def do_OPTIONS(self):
        self._send(204)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj).encode())

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}")

    # -------- routes --------

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        query = dict(p.split("=", 1) for p in self.path.split("?", 1)[1].split("&")
                     if "=" in p) if "?" in self.path else {}
        if path in ("/", "/index.html"):
            html = (WEB_DIR / "index.html").read_bytes()
            return self._send(200, html, "text/html; charset=utf-8")
        if path.startswith("/assets/"):
            name = path.rsplit("/", 1)[1]
            if name in ASSETS and (ASSETS_DIR / name).exists():
                return self._send(200, (ASSETS_DIR / name).read_bytes(), "image/png")
            return self._send(404, b"")
        if path == "/api/export":
            match = self.server.match
            if match is not None and match.result is not None:
                blob = match.replay_payload()
                if blob and blob not in [i["blob"] for i in self.server.archive.items]:
                    self.server.archive.add(match.replay_meta(), blob,
                                            match.rpl_bytes())
            data = self.server.archive.get(query.get("id"), "rpl")
            if not data:
                return self._json({"error": "nothing to export"}, 404)
            name = (query.get("id") or "match") + ".rpl"
            self.send_response(200)
            self.send_header("Content-Type", "application/gzip")
            self.send_header("Content-Disposition", f'attachment; filename="{name}"')
            self.send_header("Content-Length", str(len(data)))
            if getattr(self.server, "cors", None):
                self.send_header("Access-Control-Allow-Origin", self.server.cors)
            self.end_headers()
            return self.wfile.write(data)
        if path in ("/api/replay", "/api/replays"):
            # Fold the current match in as soon as it has a result, so it shows
            # up in the list without waiting for the next game to start.
            match = self.server.match
            if match is not None and match.result is not None:
                blob = match.replay_payload()
                if blob and blob not in [i["blob"] for i in self.server.archive.items]:
                    self.server.archive.add(match.replay_meta(), blob,
                                            match.rpl_bytes())
            if path == "/api/replays":
                return self._json({"replays": self.server.archive.listing()})
            data = self.server.archive.get(query.get("id"))
            if not data:
                return self._json({"error": "no finished match to replay"}, 404)
            return self._json({"gzip_b64": data})
        if path == "/api/bots":
            return self._json({"bots": list_bots()})
        if path == "/api/state":
            match = self.server.match
            if match is None:
                return self._json({"idle": True})
            since = int(query.get("v", -1))
            if match.version == since:
                return self._send(204)    # nothing new; keeps polling cheap
            return self._json(match.snapshot())
        return self._send(404, b"{}")

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        try:
            body = self._body()
        except Exception:
            return self._json({"error": "bad json"}, 400)

        if path == "/api/new":
            with self.server.match_lock:
                if self.server.match is not None:
                    old = self.server.match
                    if old.result is None:
                        old.result = f"abandoned at turn {old.turn}"
                    blob = old.replay_payload()
                    if blob:
                        self.server.archive.add(old.replay_meta(), blob,
                                                old.rpl_bytes())
                    old.close()
                    self.server.match = None
                try:
                    self.server.match = Match(
                        bot_name=body.get("bot", "my_bot6"),
                        seed=int(body.get("seed", 0)),
                        seat=int(body.get("seat", 0)),
                        tick_ms=int(body.get("tick_ms", 250)),
                    )
                except Exception as exc:
                    return self._json({"error": str(exc)}, 400)
            return self._json(self.server.match.snapshot())

        match = self.server.match
        if match is None:
            return self._json({"error": "no match"}, 409)

        if path == "/api/queue":
            n = match.enqueue(body.get("moves", []))
            return self._json({"queued": n})
        if path == "/api/clear":
            match.clear_queue()
            return self._json({"queued": 0})
        if path == "/api/build":
            ok, msg = match.request_build(int(body["r"]), int(body["c"]))
            return self._json({"ok": ok, "message": msg})
        if path == "/api/autopilot":
            ok, msg = match.set_autopilot(body.get("bot") or None)
            return self._json({"ok": ok, "message": msg})
        if path == "/api/resign":
            match.close()
            with self.server.match_lock:
                self.server.match = None
            return self._json({"ok": True})
        return self._send(404, b"{}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1",
                    help="bind address (use 0.0.0.0 on a VPS)")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--relay", default=None, metavar="URL",
                    help="worker mode: don't listen locally, dial out to this "
                         "relay (e.g. https://your-domain) and serve browsers "
                         "through it. The engine and bots stay on this machine.")
    ap.add_argument("--token", default=None,
                    help="shared secret matching the relay's --token")
    ap.add_argument("--replays", type=int, default=10, metavar="N",
                    help="how many finished matches to keep for replay "
                         "(default: 10, ~100 KB each)")
    ap.add_argument("--cors", default=None, metavar="ORIGIN",
                    help="allow this origin to call the API (e.g. "
                         "https://yourapp.vercel.app, or * for any). Only needed "
                         "when the frontend is hosted off-box.")
    args = ap.parse_args()

    if args.relay:
        if not args.token:
            ap.error("--relay requires --token")
        return run_worker(args.relay, args.token)

    httpd = Server((args.host, args.port), Handler)
    httpd.cors = args.cors
    httpd.archive = Archive(args.replays)
    print(f"[matchup3] serving on http://{args.host}:{args.port}/")
    print(f"[matchup3] bots: {', '.join(list_bots())}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[matchup3] shutting down")
        if httpd.match:
            httpd.match.close()


if __name__ == "__main__":
    main()
