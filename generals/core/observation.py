"""
JAX-optimized observation using NamedTuples for better performance.

Using NamedTuples avoids Python dictionary overhead and provides:
- Better JIT compilation
- Faster attribute access
- Type safety
- Pytree compatibility for JAX transformations
"""

from typing import NamedTuple
import jax.numpy as jnp


class Observation(NamedTuple):
    """
    JAX-optimized observation using NamedTuple for minimal Python overhead.
    All fields are JAX arrays for efficient vectorization and JIT compilation.
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
    owned_land_count: jnp.ndarray  # scalar int
    owned_army_count: jnp.ndarray  # scalar int
    opponent_land_count: jnp.ndarray  # scalar int
    opponent_army_count: jnp.ndarray  # scalar int
    timestep: jnp.ndarray  # scalar int
    priority: jnp.ndarray = jnp.int32(0)  # scalar int
    
    def as_tensor(self) -> jnp.ndarray:
        """
        Returns a tensor suitable for neural nets.
        Shape depends on armies shape:
        - If armies is (H, W): returns (15, H, W)
        - If armies is (N, P, H, W): returns (N, P, 15, H, W) for N envs, P players
        """
        shape = self.armies.shape
        
        # Broadcast scalar values to match spatial dimensions
        # For vectorized case: armies is (N, P, H, W), scalars are (N, P)
        # We need to expand to (N, P, H, W) then reshape for stacking
        if len(shape) == 4:  # Vectorized: (N, P, H, W)
            # Expand scalars from (N, P) to (N, P, H, W)
            owned_land = jnp.broadcast_to(
                self.owned_land_count[..., None, None], shape
            )
            owned_army = jnp.broadcast_to(
                self.owned_army_count[..., None, None], shape
            )
            opponent_land = jnp.broadcast_to(
                self.opponent_land_count[..., None, None], shape
            )
            opponent_army = jnp.broadcast_to(
                self.opponent_army_count[..., None, None], shape
            )
            timestep_broadcast = jnp.broadcast_to(
                self.timestep[..., None, None], shape
            )
            priority_broadcast = jnp.broadcast_to(
                self.priority[..., None, None], shape
            )
            
            # Stack along new axis at position 2: (N, P, 15, H, W)
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
        else:  # Non-vectorized: (H, W)
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
