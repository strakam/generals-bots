"""Coworld boundary for classic 1v1, four-player FFA and castle building."""

import jax
import jax.numpy as jnp
import numpy as np

from generals import GeneralsEnv
from generals.core import game
from generals.core.match import make_board, make_transition
from generals.modifiers import build_castles

from .protocol import PASS, VERSION

RULESET = "classic"


@jax.jit
def executed_moves(state, actions):
    """Receipts for the base moves, using the engine's actual resolution order.

    Observe each action separately, before growth or the other action can mask
    its effects.
    """
    executed = jnp.zeros((actions.shape[0],), dtype=bool)
    for player in game._determine_move_order(state, actions):
        action = actions[player]
        r, c = action[1], action[2]
        before = state.armies[r, c]
        state = game.execute_action(state, player, action)
        executed = executed.at[player].set((action[0] == 0) & (state.armies[r, c] < before))
    return executed


@jax.jit
def build_receipts(state, actions):
    """Measure construction before an opponent can capture the new castle."""
    built_state, moves = build_castles.apply_build_actions(state, actions)
    rows, cols = actions[:, 1], actions[:, 2]
    owns = state.ownership[jnp.arange(actions.shape[0]), rows, cols]
    built = (actions[:, 0] == 2) & owns & ~state.castles[rows, cols] & built_state.castles[rows, cols]
    return built, executed_moves(built_state, moves)


_build_cost_grid = jax.jit(build_castles.build_cost_grid)


class Match:
    def __init__(self, seed: int, num_players: int = 2, ruleset: str = RULESET):
        if num_players not in (2, 4):
            raise ValueError("two or four players are required")
        if ruleset not in ("classic", "build_castles"):
            raise ValueError("unknown ruleset")
        if ruleset == "build_castles" and num_players != 2:
            raise ValueError("castle-building requires two players")
        self.num_players = num_players
        self.ruleset = ruleset
        # Four spawns need a smaller separation target on the same board. The
        # generator chooses distant reachable sites and safely falls back when
        # terrain prevents satisfying every pair's target separation.
        self.env = GeneralsEnv(
            min_grid_size=18, max_grid_size=21, pad_to=21, truncation=1200,
            mountain_density_range=(0.24, 0.26), min_generals_distance=17 if num_players == 2 else 10,
            num_players=num_players, build_castles=ruleset == "build_castles", deathtouch_turn=None,
        )
        self.state = make_board(self.env, seed)
        self.transition = jax.jit(make_transition(self.env))
        self.last_move_executed = [None] * num_players
        self.last_build_executed = [None] * num_players
        # Compile before /healthz and before player deadlines begin.
        passes = jnp.array([PASS] * num_players, dtype=jnp.int32)
        jax.block_until_ready(self.transition(self.state, passes))
        jax.block_until_ready(executed_moves(self.state, passes))
        if ruleset == "build_castles":
            jax.block_until_ready(build_receipts(self.state, passes))
        for slot in range(num_players):
            jax.block_until_ready(game.get_observation(self.state, slot))
            if ruleset == "build_castles":
                jax.block_until_ready(_build_cost_grid(self.state, slot))
        self.height, self.width = self.state.armies.shape

    @property
    def turn(self):
        return int(self.state.time)

    @property
    def active_slots(self) -> list[int]:
        return np.flatnonzero(~np.asarray(self.state.eliminated)).tolist()

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
        visible_owners = np.zeros_like(kinds)
        visible = ~(np.asarray(obs.fog_cells) | np.asarray(obs.structures_in_fog))
        for owner in range(self.num_players):
            visible_owners[np.asarray(self.state.ownership[owner]) & visible] = owner + 1
        observation = {
            "type": "observation",
            "protocol_version": VERSION,
            "slot": slot,
            "turn": self.turn,
            "last_move_executed": self.last_move_executed[slot],
            "last_build_executed": self.last_build_executed[slot],
            "ruleset": self.ruleset,
            "eliminated": bool(self.state.eliminated[slot]),
            "height": self.height,
            "width": self.width,
            "my_land": int(obs.owned_land_count),
            "my_army": int(obs.owned_army_count),
            "opp_land": int(obs.opponent_land_count),
            "opp_army": int(obs.opponent_army_count),
            "type_grid": kinds.tolist(),
            "owner_grid": owners.tolist(),
            "visible_owner_grid": visible_owners.tolist(),
            "army_grid": np.asarray(obs.armies).tolist(),
        }
        if self.ruleset == "build_castles":
            eligible = np.asarray(obs.owned_cells) & (kinds == 1)
            costs = np.asarray(_build_cost_grid(self.state, slot))
            observation["build_cost_grid"] = np.where(eligible, costs, 0).tolist()
        return observation

    def frame(self) -> dict:
        s = self.state
        kinds = np.ones((self.height, self.width), dtype=np.int32)
        for mask, value in ((s.mountains, 2), (s.castles, 3), (s.generals, 4)):
            kinds[np.asarray(mask)] = value
        owners = np.zeros_like(kinds)
        for slot in range(self.num_players):
            owners[np.asarray(s.ownership[slot])] = slot + 1
        info = game.get_info(s)
        return {
            "turn": self.turn,
            "type_grid": kinds.tolist(),
            "owner_grid": owners.tolist(),
            "army_grid": np.asarray(s.armies).tolist(),
            "army": np.asarray(info.army).tolist(),
            "land": np.asarray(info.land).tolist(),
            "eliminated": np.asarray(s.eliminated).tolist(),
        }

    def advance(self, actions: list[list[int]]) -> int:
        if len(actions) != self.num_players:
            raise ValueError("one action per player is required")
        allowed = (0, 1, 2) if self.ruleset == "build_castles" else (0, 1)
        if any(action[0] not in allowed for action in actions):
            raise ValueError("classic rules accept only moves and passes")
        batch = jnp.array(actions, dtype=jnp.int32)
        if self.ruleset == "build_castles":
            built, executed = build_receipts(self.state, batch)
            self.last_build_executed = [bool(built[s]) if actions[s][0] == 2 else None for s in range(self.num_players)]
        else:
            executed = executed_moves(self.state, batch)
        self.state, info = self.transition(self.state, batch)
        self.last_move_executed = [bool(executed[s]) if actions[s][0] == 0 else None for s in range(self.num_players)]
        return int(info.winner) if bool(info.is_done) else -1

    def forfeit(self, slots: list[int]) -> int:
        """Eliminate forfeiting players; their intact armies become neutral.

        A forfeited general becomes a neutral castle, like a captured general.
        Other players keep playing until at most one remains. Unlike capture,
        no opponent receives the forfeiter's territory or a free army transfer.
        """
        if any(type(slot) is not int or not 0 <= slot < self.num_players for slot in slots):
            raise ValueError("invalid forfeiting player")
        s = self.state
        ownership = np.asarray(s.ownership).copy()
        eliminated = np.asarray(s.eliminated).copy()
        neutralized = np.zeros(s.armies.shape, dtype=bool)
        for slot in slots:
            neutralized |= ownership[slot]
            ownership[slot] = False
            eliminated[slot] = True
        active = np.flatnonzero(~eliminated)
        winner = int(active[0]) if len(active) == 1 else -1
        self.state = s._replace(
            ownership=jnp.asarray(ownership),
            ownership_neutral=s.ownership_neutral | jnp.asarray(neutralized),
            castles=s.castles | (s.generals & jnp.asarray(neutralized)),
            generals=s.generals & ~jnp.asarray(neutralized),
            eliminated=jnp.asarray(eliminated),
            winner=jnp.int32(winner),
        )
        return winner
