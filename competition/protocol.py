"""
Stdin/stdout wire protocol for language-agnostic generals agents.

This module is **engine-side only** -- it emits frames to agents and parses
their replies. The inverse operations (decoding observations, encoding
actions) belong to the agent and live in each language starter under
`starters/<lang>/`.

Frame format (all integers, line-delimited, ASCII):

  Handshake (once, engine -> agent):
    <player_id> <H> <W>

  Per turn (engine -> agent):
    <turn> <my_land> <my_army> <opp_land> <opp_army>
    H lines of W ints   # type:  0=fog 1=plain 2=mountain 3=city 4=general 5=structure-in-fog
    H lines of W ints   # owner: 0=neutral/unknown 1=me 2=opp
    H lines of W ints   # army:  0 if not visible

  Per turn (agent -> engine):
    <pass> <row> <col> <dir> <split>
        pass:  1 to skip, 0 to move
        dir:   0=up 1=down 2=left 3=right
        split: 0=move all-but-one, 1=move half

  Game end: the engine simply closes the agent's stdin. The agent should
  treat EOF on stdin as "game over, exit cleanly". No explicit signal.

Owner is perspective-relative -- agents never need to know whether they're
player 0 or 1.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from generals.core.observation import Observation


TYPE_FOG = 0
TYPE_PLAIN = 1
TYPE_MOUNTAIN = 2
TYPE_CITY = 3
TYPE_GENERAL = 4
TYPE_STRUCTURE_IN_FOG = 5

OWNER_NEUTRAL = 0
OWNER_ME = 1
OWNER_OPP = 2


def encode_handshake(player_id: int, H: int, W: int) -> str:
    return f"{player_id} {H} {W}\n"


def encode_observation(obs: Observation) -> str:
    """Encode a perspective-baked Observation into the wire-protocol frame."""
    armies = np.asarray(obs.armies, dtype=np.int32)
    H, W = armies.shape

    fog = np.asarray(obs.fog_cells, dtype=bool)
    structures_fog = np.asarray(obs.structures_in_fog, dtype=bool)
    mountains = np.asarray(obs.mountains, dtype=bool)
    cities = np.asarray(obs.cities, dtype=bool)
    generals = np.asarray(obs.generals, dtype=bool)

    type_grid = np.full((H, W), TYPE_PLAIN, dtype=np.int32)
    type_grid[fog] = TYPE_FOG
    type_grid[structures_fog] = TYPE_STRUCTURE_IN_FOG
    type_grid[mountains] = TYPE_MOUNTAIN
    type_grid[cities] = TYPE_CITY
    type_grid[generals] = TYPE_GENERAL

    owned = np.asarray(obs.owned_cells, dtype=bool)
    opp = np.asarray(obs.opponent_cells, dtype=bool)
    owner_grid = np.full((H, W), OWNER_NEUTRAL, dtype=np.int32)
    owner_grid[owned] = OWNER_ME
    owner_grid[opp] = OWNER_OPP

    lines = [
        f"{int(obs.timestep)} {int(obs.owned_land_count)} {int(obs.owned_army_count)} "
        f"{int(obs.opponent_land_count)} {int(obs.opponent_army_count)}"
    ]
    for grid in (type_grid, owner_grid, armies):
        for r in range(H):
            lines.append(" ".join(str(int(x)) for x in grid[r]))
    return "\n".join(lines) + "\n"


def decode_action(line: str) -> jnp.ndarray:
    """Parse one line into a 5-element int32 action array."""
    parts = [int(x) for x in line.split()]
    return jnp.array(parts, dtype=jnp.int32)
