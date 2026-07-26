"""
Public relay for human-vs-bot matches.

Runs on the VPS. Holds **no game logic and no engine** — it serves the frontend
and brokers messages between browsers and a worker (`matchup3.py --relay`)
running on your laptop, which owns the engine and the bot subprocess.

    python competition/relay.py --host 0.0.0.0 --port 8080 --token SECRET

The laptop dials out, so nothing needs to be reachable behind your NAT:

    python competition/matchup3.py --relay https://your-domain --token SECRET

When no worker has checked in recently the site reports "no bots live" and the
Start button is disabled.

Why polling and not websockets: stdlib only, so the VPS needs nothing beyond
python3 — no pip install at all, since the engine never runs here.

Message flow
------------
    browser  --POST /api/new|queue|build--> relay   (queued as a command)
    laptop   --POST /agent/sync {snapshot}-> relay  (long-polls, drains commands)
    browser  --GET  /api/state?v=N -------> relay   (serves the cached snapshot)
"""
import argparse
import json
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

WEB_DIR = Path(__file__).resolve().parent / "web"
# Same artwork the pygame GUI uses; served straight off disk, no engine needed.
ASSETS_DIR = Path(__file__).resolve().parent.parent / "generals" / "assets" / "images"
ASSETS = {"crownie.png", "citie.png", "mountainie.png"}

# How long /agent/sync blocks waiting for work before returning empty. Keeps the
# laptop's idle traffic near zero while making Start feel instant.
LONG_POLL = 20.0
# Grace period after a worker's last check-in. Must exceed LONG_POLL: an idle
# worker only re-syncs when its held request returns, so a shorter window would
# declare a perfectly healthy laptop dead between polls.
LIVE_TIMEOUT = LONG_POLL + 10.0


class Hub:
    """Shared state between browsers and the worker. Pure message passing."""

    def __init__(self, token):
        self.token = token
        self.lock = threading.Condition()
        self.commands = deque()
        self.snapshot = None
        self.bots = []
        self.last_seen = 0.0
        self.in_flight = 0
        self.replays = []        # [{id, meta, blob}], newest first
        self._next = 1
        self.replay_limit = 10

    def live(self):
        # A worker currently parked in a long-poll is connected by definition;
        # the timestamp only covers the gap between polls.
        return self.in_flight > 0 or (time.time() - self.last_seen) < LIVE_TIMEOUT

    def push_command(self, cmd):
        with self.lock:
            self.commands.append(cmd)
            self.lock.notify_all()

    def drain(self, timeout):
        """Worker side: wait briefly for commands, then hand over whatever queued."""
        deadline = time.time() + timeout
        with self.lock:
            self.in_flight += 1
            try:
                while not self.commands:
                    remaining = deadline - time.time()
                    if remaining <= 0:
                        return []
                    self.lock.wait(remaining)
                out = list(self.commands)
                self.commands.clear()
                return out
            finally:
                self.in_flight -= 1
                self.last_seen = time.time()

    def set_snapshot(self, snap, bots):
        with self.lock:
            if snap is not None:
                self.snapshot = snap
            self.bots = bots or self.bots
            self.last_seen = time.time()

    def add_replay(self, meta, blob, rpl_b64=None):
        """Archive a finished match, newest first, capped at replay_limit."""
        with self.lock:
            rid = f"m{self._next:04d}"
            self._next += 1
            self.replays.insert(0, {"id": rid, "meta": meta or {}, "blob": blob,
                                    "rpl": rpl_b64})
            del self.replays[self.replay_limit:]

    def replay_listing(self):
        with self.lock:
            return [{"id": it["id"], **it["meta"],
                     "kb": len(it["blob"]) // 1024} for it in self.replays]

    def replay_blob(self, rid=None, key="blob"):
        with self.lock:
            if not self.replays:
                return None
            if rid is None:
                return self.replays[0][key]
            for it in self.replays:
                if it["id"] == rid:
                    return it[key]
            return None

    def clear_match(self):
        # Only the live board is cleared; archived replays survive a new match.
        with self.lock:
            self.snapshot = None


class Server(ThreadingHTTPServer):
    daemon_threads = True
    hub: Hub = None
    cors: str = None


class Handler(BaseHTTPRequestHandler):
    server_version = "generals-relay"

    def log_message(self, *args):
        pass

    def _send(self, code, body=b"", ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        if self.server.cors:
            self.send_header("Access-Control-Allow-Origin", self.server.cors)
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()
        if body:
            self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj).encode())

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}")

    def do_OPTIONS(self):
        self._send(204)

    def do_GET(self):
        hub = self.server.hub
        path = self.path.split("?", 1)[0]
        query = dict(p.split("=", 1) for p in self.path.split("?", 1)[1].split("&")
                     if "=" in p) if "?" in self.path else {}

        if path in ("/", "/index.html"):
            return self._send(200, (WEB_DIR / "index.html").read_bytes(),
                              "text/html; charset=utf-8")
        if path.startswith("/assets/"):
            name = path.rsplit("/", 1)[1]
            if name in ASSETS and (ASSETS_DIR / name).exists():
                return self._send(200, (ASSETS_DIR / name).read_bytes(), "image/png")
            return self._send(404, b"")
        if path == "/api/bots":
            live = hub.live()
            return self._json({"bots": hub.bots if live else [], "live": live})
        if path == "/api/export":
            b64 = hub.replay_blob(query.get("id"), "rpl")
            if not b64:
                return self._json({"error": "nothing to export"}, 404)
            import base64 as _b64
            data = _b64.b64decode(b64)
            name = (query.get("id") or "match") + ".rpl"
            self.send_response(200)
            self.send_header("Content-Type", "application/gzip")
            self.send_header("Content-Disposition", f'attachment; filename="{name}"')
            self.send_header("Content-Length", str(len(data)))
            if self.server.cors:
                self.send_header("Access-Control-Allow-Origin", self.server.cors)
            self.end_headers()
            return self.wfile.write(data)
        if path == "/api/replays":
            return self._json({"replays": hub.replay_listing()})
        if path == "/api/replay":
            blob = hub.replay_blob(query.get("id"))
            if not blob:
                return self._json({"error": "no finished match to replay"}, 404)
            return self._json({"gzip_b64": blob})
        if path == "/api/state":
            live = hub.live()
            snap = hub.snapshot
            if snap is None:
                return self._json({"idle": True, "live": live})
            if int(query.get("v", -1)) == snap.get("version"):
                return self._send(204)
            return self._json({**snap, "live": live})
        return self._send(404, b"{}")

    def do_POST(self):
        hub = self.server.hub
        path = self.path.split("?", 1)[0]
        try:
            body = self._body()
        except Exception:
            return self._json({"error": "bad json"}, 400)

        # ---- worker side ----
        if path == "/agent/sync":
            if body.get("token") != hub.token:
                return self._json({"error": "bad token"}, 403)
            if body.get("ended"):
                hub.clear_match()
            if body.get("replay"):
                hub.add_replay(body.get("replay_meta"), body["replay"],
                               body.get("rpl_b64"))
            hub.set_snapshot(body.get("snapshot"), body.get("bots"))
            wait = LONG_POLL if body.get("idle") else 0.0
            return self._json({"commands": hub.drain(wait)})

        # ---- browser side ----
        if not hub.live():
            return self._json({"error": "no bot worker connected"}, 503)
        if path in ("/api/new", "/api/queue", "/api/clear", "/api/build",
                    "/api/autopilot", "/api/resign"):
            if path == "/api/new":
                # A fresh game invalidates the cached board immediately, so the
                # client does not briefly render the previous match.
                hub.clear_match()
            hub.push_command({"op": path.rsplit("/", 1)[1], "args": body})
            return self._json({"ok": True})
        return self._send(404, b"{}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--token", required=True,
                    help="shared secret the laptop worker must present")
    ap.add_argument("--replays", type=int, default=10, metavar="N",
                    help="how many finished matches to keep for replay "
                         "(default: 10, ~100 KB each)")
    ap.add_argument("--cors", default=None,
                    help="allow this origin (only if the frontend is hosted "
                         "elsewhere, e.g. Vercel)")
    args = ap.parse_args()

    httpd = Server((args.host, args.port), Handler)
    httpd.hub = Hub(args.token)
    httpd.hub.replay_limit = max(1, args.replays)
    httpd.cors = args.cors
    print(f"[relay] serving on http://{args.host}:{args.port}/  (no engine here)")
    print("[relay] waiting for a worker: matchup3.py --relay <url> --token ...")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[relay] shutting down")


if __name__ == "__main__":
    main()
