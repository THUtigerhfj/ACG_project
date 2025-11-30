"""Boundary (ghost) particles for proper density estimation at walls.

These static particles are placed just outside the container walls to provide
neighbor support for fluid particles near boundaries, preventing the density
deficit that causes boundary particles to have artificially low densities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import warp as wp


@dataclass
class BoundaryParticles:
    """Static boundary particles positioned on container walls."""
    
    positions: wp.array
    count: int
    device: str = "cuda"

    @staticmethod
    def create_box_boundary(
        half_extents: Tuple[float, float, float],
        translation: Tuple[float, float, float],
        spacing: float,
        layers: int = 2,
        device: str = "cuda",
    ) -> "BoundaryParticles":
        """Create boundary particles on the inside walls of a box container.
        
        Args:
            half_extents: Half-size of the box in each dimension
            translation: Center position of the box
            spacing: Distance between boundary particles (should match fluid particle spacing)
            layers: Number of particle layers on each wall (typically 1-2)
            device: Compute device
            
        Returns:
            BoundaryParticles instance with positions on all 6 faces
        """
        hx, hy, hz = half_extents
        tx, ty, tz = translation
        
        all_positions = []
        
        # For each layer, offset inward from the wall
        for layer in range(layers):
            offset = spacing * (layer + 0.5)  # Offset from wall surface
            
            # Generate particles on each face
            # -X face (left wall)
            xs_neg = np.array([-hx + offset])
            ys = np.arange(-hy + offset, hy - offset + spacing * 0.5, spacing)
            zs = np.arange(-hz + offset, hz - offset + spacing * 0.5, spacing)
            for y in ys:
                for z in zs:
                    all_positions.append([-hx + offset + tx, y + ty, z + tz])
            
            # +X face (right wall)
            for y in ys:
                for z in zs:
                    all_positions.append([hx - offset + tx, y + ty, z + tz])
            
            # -Y face (bottom wall)
            xs = np.arange(-hx + offset, hx - offset + spacing * 0.5, spacing)
            for x in xs:
                for z in zs:
                    all_positions.append([x + tx, -hy + offset + ty, z + tz])
            
            # +Y face (top wall) - usually open for fluid, but include for completeness
            # Comment out if you want an open-top container
            # for x in xs:
            #     for z in zs:
            #         all_positions.append([x + tx, hy - offset + ty, z + tz])
            
            # -Z face (back wall)
            for x in xs:
                for y in ys:
                    all_positions.append([x + tx, y + ty, -hz + offset + tz])
            
            # +Z face (front wall)
            for x in xs:
                for y in ys:
                    all_positions.append([x + tx, y + ty, hz - offset + tz])
        
        positions_np = np.array(all_positions, dtype=np.float32)
        positions_wp = wp.array(positions_np, dtype=wp.vec3, device=device)
        
        return BoundaryParticles(
            positions=positions_wp,
            count=len(all_positions),
            device=device,
        )


def compute_boundary_contribution(
    boundary: BoundaryParticles,
    mass: float,
    h: float,
    poly6_coeff: float,
) -> Tuple[wp.array, float]:
    """Precompute the density contribution from boundary particles.
    
    Since boundary particles are static and uniformly spaced, we can
    precompute a per-boundary-particle density contribution that gets
    added to fluid particle density calculations.
    
    Returns:
        Tuple of (boundary positions array, mass per boundary particle)
    """
    # For simplicity, each boundary particle contributes the same mass
    # as a fluid particle. This can be tuned.
    return boundary.positions, mass
