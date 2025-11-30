"""High level simulation loop hooking all Warp kernels together."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import warp as wp

from src.kernels.fluids import (
    copy_vec3_array_kernel,
    predict_positions_kernel,
    update_velocity_kernel,
    xsph_kernel,
)
from src.sim.collision import ContainerState, container_collision_kernel
from src.sim.grid import NeighborGrid
from src.sim.particles import ParticleState, initialize_lattice
from src.sim.pressure import PressureSolver
from src.utils.config import ProjectConfig


@dataclass
class SimulationState:
    particles: ParticleState
    container: ContainerState


class PBFSimulator:
    def __init__(self, config: ProjectConfig, state: SimulationState, device: str = "cuda") -> None:
        self.config = config
        self.state = state
        self.device = device
        sim_cfg = config.simulation
        self.gravity = wp.vec3(*sim_cfg.gravity)
        self.grid = NeighborGrid(sim_cfg.smoothing_length, sim_cfg.max_particles, device=device)
        self.pressure = PressureSolver(
            rest_density=sim_cfg.rest_density,
            smoothing_length=sim_cfg.smoothing_length,
            particle_mass=sim_cfg.particle_mass,
            iterations=sim_cfg.pressure_iterations,
            device=device,
        )
        self.mass = sim_cfg.particle_mass
        self.h = sim_cfg.smoothing_length
        self.xsph_coeff = sim_cfg.xsph_coefficient
        self.substeps = sim_cfg.substeps
        self.dt = sim_cfg.dt
        self.poly6_coeff, _ = (self.pressure.poly6_coeff, self.pressure.spiky_coeff)

    def step(self) -> None:
        for _ in range(self.substeps):
            self._substep()

    def _substep(self) -> None:
        particles = self.state.particles
        wp.launch(
            predict_positions_kernel,
            dim=particles.count,
            inputs=[particles.positions, particles.prev_positions, particles.velocities, self.dt, self.gravity],
            device=self.device,
        )

        self.grid.build(particles.positions)
        self.pressure.solve(particles, self.grid.id) # {density, lambda, delta, apply_delta}_kernel inside

        wp.launch(
            update_velocity_kernel,
            dim=particles.count,
            inputs=[particles.positions, particles.prev_positions, particles.velocities, 1.0 / self.dt],
            device=self.device,
        )

        # Rebuild grid with updated positions for XSPH
        self.grid.build(particles.positions)

        # XSPH: copy velocities to temp, compute XSPH into velocities
        wp.launch(
            copy_vec3_array_kernel,
            dim=particles.count,
            inputs=[particles.velocities, particles.velocities_temp],
            device=self.device,
        )
        wp.launch(
            xsph_kernel,
            dim=particles.count,
            inputs=[
                particles.positions,
                particles.velocities_temp,  # read from copy
                particles.velocities,       # write result back to main
                self.h,
                self.xsph_coeff,
                self.pressure.poly6_coeff,
                self.grid.id,
            ],
            device=self.device,
        )

        wp.launch(
            container_collision_kernel,
            dim=particles.count,
            inputs=[
                particles.positions,
                particles.velocities,
                self.state.container.translation_wp(),
                self.state.container.half_extents_wp(),
            ],
            device=self.device,
        )


def build_simulation_state(config: ProjectConfig, device: str = "cuda") -> SimulationState:
    """Construct particle and container state from config presets."""

    lattice = config.simulation.initial_lattice
    particles = initialize_lattice(
        min_corner=np.array(lattice.min_corner, dtype=float),
        max_corner=np.array(lattice.max_corner, dtype=float),
        spacing=lattice.spacing,
        device=device,
    )
    container_cfg = config.container
    container = ContainerState(
        half_extents=tuple(container_cfg.half_extents),
        translation=tuple(container_cfg.translation),
        wall_thickness=container_cfg.wall_thickness,
        device=device,
    )
    return SimulationState(particles=particles, container=container)