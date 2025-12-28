"""Game observation for JAX environment."""
from typing import NamedTuple

import jax.numpy as jnp


class Observation(NamedTuple):
    """Player observation with fog of war applied."""

    armies: jnp.ndarray  # (H, W) army counts
    generals: jnp.ndarray  # (H, W) general positions
    cities: jnp.ndarray  # (H, W) city positions
    mountains: jnp.ndarray  # (H, W) mountain positions
    neutral_cells: jnp.ndarray  # (H, W) neutral cells
    owned_cells: jnp.ndarray  # (H, W) cells owned by player
    opponent_cells: jnp.ndarray  # (H, W) cells owned by opponent
    fog_cells: jnp.ndarray  # (H, W) unexplored cells
    structures_in_fog: jnp.ndarray  # (H, W) cities/mountains in fog
    owned_land_count: jnp.ndarray  # scalar
    owned_army_count: jnp.ndarray  # scalar
    opponent_land_count: jnp.ndarray  # scalar
    opponent_army_count: jnp.ndarray  # scalar
    timestep: jnp.ndarray  # scalar
    priority: jnp.ndarray = jnp.int32(0)  # scalar

    def as_tensor(self) -> jnp.ndarray:
        """Convert to (15, H, W) tensor for neural networks."""
        shape = self.armies.shape

        if len(shape) == 4:  # Vectorized: (N, P, H, W)
            owned_land = jnp.broadcast_to(self.owned_land_count[..., None, None], shape)
            owned_army = jnp.broadcast_to(self.owned_army_count[..., None, None], shape)
            opponent_land = jnp.broadcast_to(self.opponent_land_count[..., None, None], shape)
            opponent_army = jnp.broadcast_to(self.opponent_army_count[..., None, None], shape)
            timestep_broadcast = jnp.broadcast_to(self.timestep[..., None, None], shape)
            priority_broadcast = jnp.broadcast_to(self.priority[..., None, None], shape)

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
                    priority_broadcast,
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
                    jnp.ones(shape, dtype=jnp.int32) * self.priority,
                ],
                axis=0,
            )
