"""Small castle-building baseline: fund an opening castle, then expand."""

import asyncio
import json
import os
import sys
from types import SimpleNamespace

from websockets.asyncio.client import connect

from competition.agents.expander_python.agent import Agent, DIRECTIONS
from .protocol import PASS, VERSION


def choose_action(observation, expander):
    o = observation
    costs = o.get("build_cost_grid")
    if costs is not None:
        for r in range(o["height"]):
            for c in range(o["width"]):
                if costs[r][c] > 0 and o["army_grid"][r][c] >= costs[r][c]:
                    return [2, r, c, 0, 0]
        # A deliberate opening demonstrates building even against another
        # baseline that would otherwise spread every newly grown army.
        if o["turn"] <= 102:
            for r in range(o["height"]):
                for c in range(o["width"]):
                    if o["owner_grid"][r][c] != 1 or o["type_grid"][r][c] != 4:
                        continue
                    if o["army_grid"][r][c] < 50:
                        return PASS.copy()
                    for direction, (dr, dc) in enumerate(DIRECTIONS):
                        nr, nc = r + dr, c + dc
                        if (0 <= nr < o["height"] and 0 <= nc < o["width"]
                                and o["type_grid"][nr][nc] == 1
                                and o["owner_grid"][nr][nc] in (0, 1)
                                and o["army_grid"][nr][nc] == 0):
                            return [0, r, c, direction, 0]
    return list(expander.act(SimpleNamespace(H=o["height"], W=o["width"], **o)))


async def play(url):
    expander = None
    async with connect(url, ping_timeout=None, max_size=128 * 1024, open_timeout=30) as ws:
        async for raw in ws:
            message = json.loads(raw)
            if message.get("type") == "hello":
                if message.get("protocol_version") != VERSION:
                    raise RuntimeError("unsupported protocol version")
                expander = Agent(message["slot"], message["height"], message["width"])
            elif message.get("type") == "observation":
                if message.get("eliminated"):
                    continue
                if expander is None:
                    raise RuntimeError("missing handshake")
                await ws.send(json.dumps({"type": "action", "turn": message["turn"],
                                          "action": choose_action(message, expander)}))
            elif message.get("type") == "final":
                return
            elif message.get("type") in ("failure", "error"):
                raise RuntimeError("game rejected the session")


def main():
    try:
        asyncio.run(play(os.environ["COWORLD_PLAYER_WS_URL"]))
    except Exception as exc:
        print(f"[builder-player] session failed ({type(exc).__name__})", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
