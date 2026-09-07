"""
Game observation for JAX environment.

This module defines the Observation class that represents what a player can see
during a game. Observations include fog of war - players can only see cells
within a 3x3 radius of cells they (or, in team games, a teammate) own.
"""
from typing import NamedTuple

import jax.numpy as jnp


class Observation(NamedTuple):
    """
    Player observation with fog of war applied.

    All spatial fields have shape (H, W) where H and W are the grid dimensions.
    Boolean masks use True to indicate presence. Army counts are integers.

    Attributes:
        armies: Army counts in visible cells (0 in fog).
        generals: Boolean mask of visible general positions.
        castles: Boolean mask of visible castle positions.
        mountains: Boolean mask of visible mountain positions.
        neutral_cells: Boolean mask of visible neutral (unowned) cells.
        owned_cells: Boolean mask of cells owned by this player.
        opponent_cells: Boolean mask of visible cells owned by any enemy team.
        fog_cells: Boolean mask of fog cells (not visible, no structure).
        structures_in_fog: Boolean mask of castles/mountains in fog (visible as obstacles).
        owned_land_count: Scalar, total number of cells owned by this player.
        owned_army_count: Scalar, total army count across all owned cells.
        opponent_land_count: Scalar, enemy teams' total cell count.
        opponent_army_count: Scalar, enemy teams' total army count.
        timestep: Scalar, current game step (0-indexed).
        allied_cells: Boolean mask of visible cells owned by a teammate (not
            self). All-False in 1v1 and free-for-all, where nobody has a teammate.
        allied_land_count: Scalar, teammates' total cell count (excluding self).
        allied_army_count: Scalar, teammates' total army count (excluding self).

    The three allied fields come last and default to None so that code
    building observations for 1v1 by hand keeps working unchanged.
    """

    armies: jnp.ndarray
    generals: jnp.ndarray
    castles: jnp.ndarray
    mountains: jnp.ndarray
    neutral_cells: jnp.ndarray
    owned_cells: jnp.ndarray
    opponent_cells: jnp.ndarray
    fog_cells: jnp.ndarray
    structures_in_fog: jnp.ndarray
    owned_land_count: jnp.ndarray
    owned_army_count: jnp.ndarray
    opponent_land_count: jnp.ndarray
    opponent_army_count: jnp.ndarray
    timestep: jnp.ndarray
    allied_cells: jnp.ndarray = None
    allied_land_count: jnp.ndarray = None
    allied_army_count: jnp.ndarray = None

    @property
    def cities(self):
        """Deprecated alias — castles were renamed from cities."""
        return self.castles

    def as_tensor(self, include_allied: bool = False) -> jnp.ndarray:
        """
        Convert observation to a tensor for neural networks.

        Returns:
            For single observations: (14, H, W) tensor — (17, H, W) with
            include_allied=True. For vectorized observations: stacked along axis 2.

        The channels are ordered as:
            0: armies, 1: generals, 2: castles, 3: mountains, 4: neutral_cells,
            5: owned_cells, 6: opponent_cells, 7: fog_cells, 8: structures_in_fog,
            9: owned_land_count, 10: owned_army_count, 11: opponent_land_count,
            12: opponent_army_count, 13: timestep
        and, with include_allied=True (team play):
            14: allied_cells, 15: allied_land_count, 16: allied_army_count
        """
        shape = self.armies.shape
        scalars = [self.owned_land_count, self.owned_army_count,
                   self.opponent_land_count, self.opponent_army_count, self.timestep]
        planes = [self.armies, self.generals, self.castles, self.mountains, self.neutral_cells,
                  self.owned_cells, self.opponent_cells, self.fog_cells, self.structures_in_fog]
        allied_planes = [self.allied_cells] if include_allied else []
        allied_scalars = [self.allied_land_count, self.allied_army_count] if include_allied else []

        if len(shape) == 4:  # Vectorized: (N, P, H, W)
            broadcast = lambda s: jnp.broadcast_to(s[..., None, None], shape)
            return jnp.stack(
                planes + [broadcast(s) for s in scalars] + allied_planes + [broadcast(s) for s in allied_scalars],
                axis=2,
            )
        else:  # Single observation: (H, W)
            ones = jnp.ones(shape, dtype=jnp.int32)
            return jnp.stack(
                planes + [ones * s for s in scalars] + allied_planes + [ones * s for s in allied_scalars],
                axis=0,
            )
