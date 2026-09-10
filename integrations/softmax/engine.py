"""The Coworld boundary around the unchanged competition rules."""

import jax
import jax.numpy as jnp
import numpy as np

from generals import GeneralsEnv
from generals.core import game
from generals.core.match import make_board, make_transition

from .protocol import PASS, VERSION


class Match:
    def __init__(self, seed: int):
        self.env = GeneralsEnv(mode="competition")
        self.state = make_board(self.env, seed)
        self.transition = jax.jit(make_transition(self.env))
        # Compile before /healthz and before player deadlines begin.
        jax.block_until_ready(self.transition(self.state, jnp.array([PASS, PASS], dtype=jnp.int32)))
        for slot in range(2):
            jax.block_until_ready(game.get_observation(self.state, slot))
        self.height, self.width = self.state.armies.shape

    @property
    def turn(self):
        return int(self.state.time)

    def observation(self, slot: int) -> dict:
        obs = game.get_observation(self.state, slot)
        kinds = np.ones((self.height, self.width), dtype=np.int32)
        for name, value in (
            ("fog_cells", 0),
            ("structures_in_fog", 5),
            ("mountains", 2),
            ("castles", 3),
            ("generals", 4),
        ):
            kinds[np.asarray(getattr(obs, name), dtype=bool)] = value
        owners = np.zeros_like(kinds)
        owners[np.asarray(obs.owned_cells)] = 1
        owners[np.asarray(obs.opponent_cells)] = 2
        return {
            "type": "observation",
            "protocol_version": VERSION,
            "slot": slot,
            "turn": self.turn,
            "height": self.height,
            "width": self.width,
            "my_land": int(obs.owned_land_count),
            "my_army": int(obs.owned_army_count),
            "opp_land": int(obs.opponent_land_count),
            "opp_army": int(obs.opponent_army_count),
            "type_grid": kinds.tolist(),
            "owner_grid": owners.tolist(),
            "army_grid": np.asarray(obs.armies).tolist(),
        }

    def frame(self) -> dict:
        s = self.state
        kinds = np.ones((self.height, self.width), dtype=np.int32)
        for mask, value in ((s.mountains, 2), (s.castles, 3), (s.generals, 4)):
            kinds[np.asarray(mask)] = value
        owners = np.zeros_like(kinds)
        for slot in range(2):
            owners[np.asarray(s.ownership[slot])] = slot + 1
        info = game.get_info(s)
        return {
            "turn": self.turn,
            "type_grid": kinds.tolist(),
            "owner_grid": owners.tolist(),
            "army_grid": np.asarray(s.armies).tolist(),
            "army": np.asarray(info.army).tolist(),
            "land": np.asarray(info.land).tolist(),
        }

    def advance(self, actions: list[list[int]]) -> int:
        self.state, info = self.transition(self.state, jnp.array(actions, dtype=jnp.int32))
        return int(info.winner) if bool(info.is_done) else -1
