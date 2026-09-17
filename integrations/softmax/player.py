"""Bridge any competition stdio bot to Coworld, without importing the engine."""

import argparse
import asyncio
import json
import os
import sys
from contextlib import suppress
from pathlib import Path

from websockets.asyncio.client import connect

from .protocol import VERSION, stdio_frame

ROOT = Path(__file__).resolve().parents[2]


async def play(command: list[str], url: str):
    proc = None
    try:
        async with connect(url, ping_timeout=None, max_size=128 * 1024, open_timeout=30) as ws:
            async for raw in ws:
                message = json.loads(raw)
                kind = message.get("type")
                if kind == "hello":
                    if message.get("protocol_version") != VERSION:
                        raise RuntimeError("unsupported protocol version")
                    proc = await asyncio.create_subprocess_exec(
                        *command,
                        stdin=asyncio.subprocess.PIPE,
                        stdout=asyncio.subprocess.PIPE,
                        # Player stderr is a diagnostic stream; never copy the connection URL into it.
                        stderr=None,
                        limit=4096,
                    )
                    proc.stdin.write(f"{message['slot']} {message['height']} {message['width']}\n".encode())
                    await proc.stdin.drain()
                elif kind == "observation":
                    if proc is None:
                        raise RuntimeError("observation received before handshake")
                    if message.get("eliminated"):
                        continue  # Remain connected for the final result; no further action deadline.
                    # Don't queue stale observations behind a hung subprocess. Failure
                    # ends this player session; the server's timeout rules own scoring.
                    async with asyncio.timeout(message["turn_timeout_seconds"]):
                        proc.stdin.write(stdio_frame(message).encode())
                        await proc.stdin.drain()
                        line = await proc.stdout.readline()
                    if not line:
                        raise RuntimeError("bot exited before the match ended")
                    action = [int(v) for v in line.split()]
                    if len(action) != 5:
                        raise RuntimeError("bot must emit five integers")
                    await ws.send(json.dumps({"type": "action", "turn": message["turn"], "action": action}))
                elif kind == "final":
                    return
                elif kind == "failure":
                    raise RuntimeError("game failed")
                elif kind == "error":
                    print(f"[coworld-player] {message['message']}", file=sys.stderr)
    finally:
        if proc is not None:
            if proc.stdin:
                proc.stdin.close()
            try:
                async with asyncio.timeout(2):
                    await proc.wait()
            except TimeoutError:
                with suppress(ProcessLookupError):
                    proc.kill()
                await proc.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs=argparse.REMAINDER, help="bot command after --; defaults to Python Expander")
    args = parser.parse_args()
    command = args.command
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        command = [sys.executable, "-u", str(ROOT / "competition/agents/expander_python/main.py")]
    try:
        asyncio.run(play(command, os.environ["COWORLD_PLAYER_WS_URL"]))
    except Exception as exc:  # noqa: BLE001 - do not print credential-bearing connection exceptions
        print(f"[coworld-player] session failed ({type(exc).__name__})", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
