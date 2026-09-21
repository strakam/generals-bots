"""Run the repository's unchanged HunterAgent through the Coworld protocol."""

import asyncio
import json
import os
import sys

import jax
import jax.numpy as jnp
from websockets.asyncio.client import connect

from generals.agents.hunter_agent import HunterAgent
from generals.core.observation import Observation

from integrations.softmax.protocol import VERSION


def observation_from_message(message):
    """Reconstruct only the player's visible observation, with relative owners."""
    kinds = jnp.asarray(message["type_grid"], dtype=jnp.int32)
    owners = jnp.asarray(message["owner_grid"], dtype=jnp.int32)
    visible = (kinds != 0) & (kinds != 5)
    scalar = lambda name: jnp.asarray(message[name], dtype=jnp.int32)
    return Observation(
        armies=jnp.asarray(message["army_grid"], dtype=jnp.int32),
        generals=kinds == 4,
        castles=kinds == 3,
        mountains=kinds == 2,
        neutral_cells=(owners == 0) & visible & (kinds != 2),
        owned_cells=owners == 1,
        opponent_cells=owners == 2,
        fog_cells=kinds == 0,
        structures_in_fog=kinds == 5,
        owned_land_count=scalar("my_land"),
        owned_army_count=scalar("my_army"),
        opponent_land_count=scalar("opp_land"),
        opponent_army_count=scalar("opp_army"),
        timestep=scalar("turn"),
        allied_cells=jnp.zeros_like(kinds, dtype=bool),
        allied_land_count=jnp.int32(0),
        allied_army_count=jnp.int32(0),
    )


def warmup(agent, key):
    # Compile every supported rectangle BEFORE connecting. Once both players
    # connect, the first observation already has a 500 ms action deadline.
    for height in range(18, 22):
        for width in range(18, 22):
            grid = [[0] * width for _ in range(height)]
            obs = observation_from_message({
                "type_grid": grid, "owner_grid": grid, "army_grid": grid,
                "my_land": 0, "my_army": 0, "opp_land": 0, "opp_army": 0, "turn": 0,
            })
            jax.block_until_ready(agent.act(obs, key))


async def play(url, agent, key):
    async with connect(url, ping_timeout=None, max_size=128 * 1024, open_timeout=30) as ws:
        async for raw in ws:
            message = json.loads(raw)
            kind = message["type"]
            if kind == "hello":
                if message.get("protocol_version") != VERSION:
                    raise RuntimeError("unsupported protocol version")
                if not (18 <= message["height"] <= 21 and 18 <= message["width"] <= 21):
                    raise RuntimeError("unsupported map size")
                agent.reset()
            elif kind == "observation":
                action = agent.act(observation_from_message(message), key).tolist()
                await ws.send(json.dumps({"type": "action", "turn": message["turn"], "action": action}))
            elif kind == "final":
                print(json.dumps({"event": "match_finished", "result": message["result"]}), file=sys.stderr)
                return
            elif kind in {"failure", "error"}:
                raise RuntimeError("game rejected the player or failed")


def main():
    try:
        agent, key = HunterAgent(), jax.random.PRNGKey(0)
        warmup(agent, key)
        print("[hunter] compiled all supported map sizes; connecting", file=sys.stderr)
        asyncio.run(play(os.environ["COWORLD_PLAYER_WS_URL"], agent, key))
    except Exception as exc:
        # Connection exceptions can contain the credential-bearing URL.
        print(f"[hunter] session failed ({type(exc).__name__})", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
