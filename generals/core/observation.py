"""
Game observation for JAX environment.

This module defines the Observation class that represents what a player can see
during a game. Observations include fog of war — visibility extends to a 3x3
radius around any cell owned by the player OR any of their teammates.
"""
from typing import NamedTuple

import jax.numpy as jnp


class Observation(NamedTuple):
    """
    Player observation with fog of war applied.

    All spatial fields have shape (H, W). Boolean masks use True to indicate
    presence; army counts are integers.

    Attributes:
        armies: Army counts in visible cells (0 in fog).
        generals: Boolean mask of visible general positions.
        cities: Boolean mask of visible city positions.
        mountains: Boolean mask of visible mountain positions.
        neutral_cells: Boolean mask of visible neutral (unowned) cells.
        owned_cells: Boolean mask of cells owned by THIS player.
        allied_cells: Boolean mask of visible cells owned by a teammate (not self).
            Empty in free-for-all modes (no teammates).
        opponent_cells: Boolean mask of visible cells owned by an opposing team.
        fog_cells: Boolean mask of fog cells (not visible, no structure).
        structures_in_fog: Boolean mask of cities/mountains in fog (visible as obstacles).
        owned_land_count: Scalar, this player's total cells.
        owned_army_count: Scalar, this player's total army count.
        allied_land_count: Scalar, teammates' total cells (excluding self).
        allied_army_count: Scalar, teammates' total army count (excluding self).
        opponent_land_count: Scalar, opposing teams' total cells.
        opponent_army_count: Scalar, opposing teams' total army count.
        timestep: Scalar, current game step (0-indexed).
    """

    armies: jnp.ndarray
    generals: jnp.ndarray
    cities: jnp.ndarray
    mountains: jnp.ndarray
    neutral_cells: jnp.ndarray
    owned_cells: jnp.ndarray
    allied_cells: jnp.ndarray
    opponent_cells: jnp.ndarray
    fog_cells: jnp.ndarray
    structures_in_fog: jnp.ndarray
    owned_land_count: jnp.ndarray
    owned_army_count: jnp.ndarray
    allied_land_count: jnp.ndarray
    allied_army_count: jnp.ndarray
    opponent_land_count: jnp.ndarray
    opponent_army_count: jnp.ndarray
    timestep: jnp.ndarray

    def as_tensor(self) -> jnp.ndarray:
        """
        Convert observation to a tensor for neural networks.

        Returns:
            For single observations: (17, H, W) tensor.
            For vectorized observations: stacked along axis 2.

        Channel ordering:
            0: armies, 1: generals, 2: cities, 3: mountains, 4: neutral_cells,
            5: owned_cells, 6: allied_cells, 7: opponent_cells,
            8: fog_cells, 9: structures_in_fog,
            10: owned_land_count, 11: owned_army_count,
            12: allied_land_count, 13: allied_army_count,
            14: opponent_land_count, 15: opponent_army_count,
            16: timestep
        """
        shape = self.armies.shape

        if len(shape) == 4:  # Vectorized: (N_envs, N_players, H, W)
            owned_land = jnp.broadcast_to(self.owned_land_count[..., None, None], shape)
            owned_army = jnp.broadcast_to(self.owned_army_count[..., None, None], shape)
            allied_land = jnp.broadcast_to(self.allied_land_count[..., None, None], shape)
            allied_army = jnp.broadcast_to(self.allied_army_count[..., None, None], shape)
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
                    self.allied_cells,
                    self.opponent_cells,
                    self.fog_cells,
                    self.structures_in_fog,
                    owned_land,
                    owned_army,
                    allied_land,
                    allied_army,
                    opponent_land,
                    opponent_army,
                    timestep_broadcast,
                ],
                axis=2,
            )
        else:  # Single observation: (H, W)
            ones = jnp.ones(shape, dtype=jnp.int32)
            return jnp.stack(
                [
                    self.armies,
                    self.generals,
                    self.cities,
                    self.mountains,
                    self.neutral_cells,
                    self.owned_cells,
                    self.allied_cells,
                    self.opponent_cells,
                    self.fog_cells,
                    self.structures_in_fog,
                    ones * self.owned_land_count,
                    ones * self.owned_army_count,
                    ones * self.allied_land_count,
                    ones * self.allied_army_count,
                    ones * self.opponent_land_count,
                    ones * self.opponent_army_count,
                    ones * self.timestep,
                ],
                axis=0,
            )
