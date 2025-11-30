"""Host-side helpers to run the Warp PBF pressure solve."""

from __future__ import annotations

from dataclasses import dataclass

import warp as wp

from src.kernels.fluids import (
    apply_delta_kernel,
    create_kernel_params,
    delta_kernel,
    density_kernel,
    lambda_kernel,
)
from src.sim.particles import ParticleState


@dataclass
class PressureSolver:
    rest_density: float
    smoothing_length: float
    particle_mass: float
    iterations: int
    device: str = "cuda"

    def __post_init__(self) -> None:
        self.poly6_coeff, self.spiky_coeff = create_kernel_params(self.smoothing_length)

    def compute_density(self, state: ParticleState, grid_id: int) -> None:
        """Compute densities for all particles."""
        wp.launch(
            density_kernel,
            dim=state.count,
            inputs=[
                state.positions,
                state.densities,
                self.particle_mass,
                self.smoothing_length,
                self.poly6_coeff,
                grid_id,
            ],
            device=self.device,
        )

    def solve(self, state: ParticleState, grid_id: int) -> None:
        dim = state.count
        self.compute_density(state, grid_id)

        for _ in range(self.iterations):
            wp.launch(
                lambda_kernel,
                dim=dim,
                inputs=[
                    state.positions,
                    state.densities,
                    state.lambdas,
                    self.particle_mass,
                    self.rest_density,
                    self.smoothing_length,
                    self.spiky_coeff,
                    1e-6,
                    grid_id,
                ],
                device=self.device,
            )
            wp.launch(
                delta_kernel,
                dim=dim,
                inputs=[
                    state.positions,
                    state.lambdas,
                    state.delta_pos,
                    self.particle_mass,
                    self.rest_density,
                    self.smoothing_length,
                    self.spiky_coeff,
                    grid_id,
                ],
                device=self.device,
            )
            wp.launch(
                apply_delta_kernel,
                dim=dim,
                inputs=[state.positions, state.delta_pos],
                device=self.device,
            )