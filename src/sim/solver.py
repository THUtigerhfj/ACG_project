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
from src.sim.rigid_spheres import (
    NUM_SPHERES,
    RigidSphereState,
    build_rigid_spheres,
    integrate_spheres_kernel,
    sphere_sphere_collision_kernel,
    sphere_container_collision_kernel,
    sphere_water_coupling_kernel,
)
from src.utils.config import ProjectConfig


@dataclass
class SimulationState:
    particles: ParticleState
    container: ContainerState
    spheres: RigidSphereState


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

        # Coupling / contact params
        rigid_cfg = config.rigid
        self.contact_offset = rigid_cfg.contact_offset
        # If max_push <= 0, derive from smoothing length (more permissive, helps contacts stick)
        self.max_push = rigid_cfg.max_push if rigid_cfg.max_push > 0.0 else self.h
        # Allow higher impulse by default; caller can still lower in config
        self.max_impulse = rigid_cfg.max_impulse
        self.particle_radius = config.viewer.particle_radius

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

        # Keep spheres inside container (translation-only)
        wp.launch(
            sphere_container_collision_kernel,
            dim=NUM_SPHERES,
            inputs=[
                self.state.spheres.centers,
                self.state.spheres.velocities,
                self.state.container.translation_wp(),
                self.state.container.half_extents_wp(),
                self.state.spheres.radii,
                self.contact_offset,
                self.max_push,
            ],
            device=self.device,
        )

        # Sphere-sphere mutual collision (only one pair when NUM_SPHERES == 2)
        wp.launch(
            sphere_sphere_collision_kernel,
            dim=1,
            inputs=[
                self.state.spheres.centers,
                self.state.spheres.velocities,
                self.state.spheres.radii,
                self.state.spheres.inv_masses,
                self.contact_offset,
                self.max_push,
                self.max_impulse,
            ],
            device=self.device,
        )

        # Two-way coupling: particles vs spheres
        wp.launch(
            sphere_water_coupling_kernel,
            dim=particles.count,
            inputs=[
                particles.positions,
                particles.velocities,
                self.state.spheres.centers,
                self.state.spheres.velocities,
                self.state.spheres.radii,
                self.state.spheres.inv_masses,
                self.state.spheres.impulse_x,
                self.state.spheres.impulse_y,
                self.state.spheres.impulse_z,
                self.mass,
                self.particle_radius,
                self.contact_offset,
                self.max_push,
                self.max_impulse,
            ],
            device=self.device,
        )

        # Integrate spheres with accumulated impulses and gravity
        wp.launch(
            integrate_spheres_kernel,
            dim=NUM_SPHERES,
            inputs=[
                self.state.spheres.centers,
                self.state.spheres.velocities,
                self.state.spheres.inv_masses,
                self.state.spheres.impulse_x,
                self.state.spheres.impulse_y,
                self.state.spheres.impulse_z,
                self.dt,
                self.gravity,
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

    rigid_cfg = config.rigid
    spheres = build_rigid_spheres(
        centers=[s.center for s in rigid_cfg.spheres],
        velocities=[s.velocity for s in rigid_cfg.spheres],
        radii=[s.radius for s in rigid_cfg.spheres],
        densities=[s.density for s in rigid_cfg.spheres],
        device=device,
    )

    return SimulationState(particles=particles, container=container, spheres=spheres)