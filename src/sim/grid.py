"""Hash grid helper used for neighbor searches."""

from __future__ import annotations

from dataclasses import dataclass

import warp as wp


@dataclass
class NeighborGrid:
    radius: float
    max_particles: int
    device: str = "cuda"

    def __post_init__(self) -> None:
        # Fixed resolution hash grid large enough for real-time demo scenes.
        resolution = 128
        self._grid = wp.HashGrid(resolution, resolution, resolution, device=self.device)

    @property
    def id(self) -> int:
        return self._grid.id

    def build(self, positions: wp.array) -> None:
        """Rebuild the hash grid around the latest particle positions."""

        self._grid.build(points=positions, radius=self.radius)

    def query(self, position: wp.vec3) -> wp.hash_grid_query:
        return wp.hash_grid_query(self._grid.id, position, self.radius)
