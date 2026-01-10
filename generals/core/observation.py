"""
Game observation for JAX environment.

This module defines the Observation class that represents what a player can see
during a game. Observations include fog of war - players can only see cells
within a 3x3 radius of cells they own.
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
        cities: Boolean mask of visible city positions.
        mountains: Boolean mask of visible mountain positions.
        neutral_cells: Boolean mask of visible neutral (unowned) cells.
        owned_cells: Boolean mask of cells owned by this player.
        opponent_cells: Boolean mask of visible opponent cells.
        fog_cells: Boolean mask of fog cells (not visible, no structure).
        structures_in_fog: Boolean mask of cities/mountains in fog (visible as obstacles).
        owned_land_count: Scalar, total number of cells owned by this player.
        owned_army_count: Scalar, total army count across all owned cells.
        opponent_land_count: Scalar, opponent's total cell count.
        opponent_army_count: Scalar, opponent's total army count.
        timestep: Scalar, current game step (0-indexed).
    """

    armies: jnp.ndarray
    generals: jnp.ndarray
    cities: jnp.ndarray
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

    def as_tensor(self) -> jnp.ndarray:
        """
        Convert observation to a tensor for neural networks.

        Returns:
            For single observations: (14, H, W) tensor.
            For vectorized observations: stacked along axis 2.

        The channels are ordered as:
            0: armies, 1: generals, 2: cities, 3: mountains, 4: neutral_cells,
            5: owned_cells, 6: opponent_cells, 7: fog_cells, 8: structures_in_fog,
            9: owned_land_count, 10: owned_army_count, 11: opponent_land_count,
            12: opponent_army_count, 13: timestep
        """
        shape = self.armies.shape

        if len(shape) == 4:  # Vectorized: (N, P, H, W)
            owned_land = jnp.broadcast_to(self.owned_land_count[..., None, None], shape)
            owned_army = jnp.broadcast_to(self.owned_army_count[..., None, None], shape)
            opponent_land = jnp.broadcast_to(self.opponent_land_count[..., None, None], shape)
            opponent_army = jnp.broadcast_to(self.opponent_army_count[..., None, None], shape)
            timestep_broadcast = jnp.broadcast_to(self.timestep[..., None, None], shape)

            return jnp.stack(
                [
                    self.armies,
                    self.generals,
                    self.cities,
                    self.mountains,
                    self.neutral_cells,
                    self.owned_cells,
                    self.opponent_cells,
                    self.fog_cells,
                    self.structures_in_fog,
                    owned_land,
                    owned_army,
                    opponent_land,
                    opponent_army,
                    timestep_broadcast,
                ],
                axis=2,
            )
        else:  # Single observation: (H, W)
            return jnp.stack(
                [
                    self.armies,
                    self.generals,
                    self.cities,
                    self.mountains,
                    self.neutral_cells,
                    self.owned_cells,
                    self.opponent_cells,
                    self.fog_cells,
                    self.structures_in_fog,
                    jnp.ones(shape, dtype=jnp.int32) * self.owned_land_count,
                    jnp.ones(shape, dtype=jnp.int32) * self.owned_army_count,
                    jnp.ones(shape, dtype=jnp.int32) * self.opponent_land_count,
                    jnp.ones(shape, dtype=jnp.int32) * self.opponent_army_count,
                    jnp.ones(shape, dtype=jnp.int32) * self.timestep,
                ],
                axis=0,
            )
