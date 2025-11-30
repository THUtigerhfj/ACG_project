"""Particle buffer management for the Warp-based SPH solver."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import warp as wp


@dataclass
class ParticleState:
    """Struct-of-arrays particle representation stored on the device."""

    positions: wp.array
    velocities: wp.array
    velocities_temp: wp.array  # For XSPH double-buffering
    prev_positions: wp.array
    densities: wp.array
    lambdas: wp.array
    delta_pos: wp.array

    @property
    def count(self) -> int:
        return int(self.positions.shape[0])


def allocate_particle_state(count: int, device: str = "cuda") -> ParticleState:
    """Allocate empty Warp arrays for all particle attributes."""

    zeros_vec = wp.zeros(count, dtype=wp.vec3, device=device)
    zeros_scalar = wp.zeros(count, dtype=float, device=device)

    return ParticleState(
        positions=zeros_vec,
        velocities=zeros_vec,
        velocities_temp=zeros_vec,
        prev_positions=zeros_vec,
        densities=zeros_scalar,
        lambdas=zeros_scalar,
        delta_pos=zeros_vec,
    )


def initialize_lattice(
    min_corner: np.ndarray,
    max_corner: np.ndarray,
    spacing: float,
    device: str = "cuda",
) -> ParticleState:
    """Generate a simple regular lattice of particles that fills the given bounds."""

    xs = np.arange(min_corner[0], max_corner[0] + spacing * 0.5, spacing)
    ys = np.arange(min_corner[1], max_corner[1] + spacing * 0.5, spacing)
    zs = np.arange(min_corner[2], max_corner[2] + spacing * 0.5, spacing)

    grid = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1).reshape(-1, 3)
    velocities = np.zeros_like(grid)

    state = allocate_particle_state(len(grid), device=device)
    state.positions = wp.array(grid, dtype=wp.vec3, device=device)
    state.prev_positions = wp.array(grid.copy(), dtype=wp.vec3, device=device)
    state.velocities = wp.array(velocities, dtype=wp.vec3, device=device)
    state.velocities_temp = wp.array(velocities.copy(), dtype=wp.vec3, device=device)

    return state
