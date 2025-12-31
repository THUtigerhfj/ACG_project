"""Configuration helpers for the real-time SPH simulator."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import yaml


@dataclass
class LatticeConfig:
    min_corner: List[float]
    max_corner: List[float]
    spacing: float


@dataclass
class SimulationConfig:
    dt: float
    substeps: int
    pressure_iterations: int
    xsph_coefficient: float
    smoothing_length: float
    rest_density: float
    particle_mass: float
    gravity: List[float]
    max_particles: int
    initial_lattice: LatticeConfig


@dataclass
class ContainerConfig:
    half_extents: List[float]
    translation: List[float]
    wall_thickness: float


@dataclass
class RigidSphereConfig:
    center: List[float]
    velocity: List[float]
    radius: float
    density: float
    color: List[float]


@dataclass
class RigidConfig:
    spheres: List[RigidSphereConfig]
    contact_offset: float
    max_push: float
    max_impulse: float


@dataclass
class ViewerConfig:
    fps: int
    particle_radius: float
    background_color: List[float]


@dataclass
class ProjectConfig:
    simulation: SimulationConfig
    container: ContainerConfig
    rigid: RigidConfig
    viewer: ViewerConfig


def _as_lattice(data: Dict[str, Any]) -> LatticeConfig:
    return LatticeConfig(
        min_corner=data.get("min_corner", [-0.5, -0.5, -0.5]),
        max_corner=data.get("max_corner", [0.5, 0.5, 0.5]),
        spacing=float(data.get("spacing", 0.02)),
    )


def _as_sim(data: Dict[str, Any]) -> SimulationConfig:
    return SimulationConfig(
        dt=float(data.get("dt", 0.005)),
        substeps=int(data.get("substeps", 3)),
        pressure_iterations=int(data.get("pressure_iterations", 6)),
        xsph_coefficient=float(data.get("xsph_coefficient", 0.1)),
        smoothing_length=float(data.get("smoothing_length", 0.04)),
        rest_density=float(data.get("rest_density", 1000.0)),
        particle_mass=float(data.get("particle_mass", 0.02)),
        gravity=list(data.get("gravity", [0.0, -9.81, 0.0])),
        max_particles=int(data.get("max_particles", 20000)),
        initial_lattice=_as_lattice(data.get("initial_lattice", {})),
    )


def _as_container(data: Dict[str, Any]) -> ContainerConfig:
    return ContainerConfig(
        half_extents=list(data.get("half_extents", [0.5, 0.5, 0.5])),
        translation=list(data.get("translation", [0.0, 0.0, 0.0])),
        wall_thickness=float(data.get("wall_thickness", 0.02)),
    )


def _as_rigid(data: Dict[str, Any]) -> RigidConfig:
    raw_spheres = data.get("spheres", [])
    if not raw_spheres:
        raw_spheres = [
            {"center": [0.0, 0.0, 0.0], "velocity": [0.0, 0.0, 0.0]},
            {"center": [0.0, 0.0, 0.0], "velocity": [0.0, 0.0, 0.0]},
        ]
    spheres: List[RigidSphereConfig] = []
    default_colors = ([0.9, 0.35, 0.2], [0.2, 0.9, 0.35])
    for idx, s in enumerate(raw_spheres):
        fallback_color = default_colors[idx % len(default_colors)]
        spheres.append(
            RigidSphereConfig(
                center=list(s.get("center", [0.0, 0.0, 0.0])),
                velocity=list(s.get("velocity", [0.0, 0.0, 0.0])),
                radius=float(s.get("radius", 0.05)),
                density=float(s.get("density", 1000.0)),
                color=list(s.get("color", fallback_color)),
            )
        )

    # Ensure exactly two spheres to match kernels.
    if len(spheres) != 2:
        raise ValueError("rigid.spheres must define exactly two spheres")

    return RigidConfig(
        spheres=spheres,
        contact_offset=float(data.get("contact_offset", 0.008)),
        max_push=float(data.get("max_push", -1.0)),  # -1 -> derive from smoothing length in solver
        max_impulse=float(data.get("max_impulse", 2.0)),
    )


def _as_viewer(data: Dict[str, Any]) -> ViewerConfig:
    return ViewerConfig(
        fps=int(data.get("fps", 60)),
        particle_radius=float(data.get("particle_radius", 0.01)),
        background_color=list(data.get("background_color", [0.02, 0.02, 0.03])),
    )


def load_config(path: str | Path) -> ProjectConfig:
    """Parse a YAML config file into strongly typed dataclasses."""

    with open(Path(path), "r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)

    simulation = _as_sim(raw.get("simulation", {}))
    container = _as_container(raw.get("container", {}))
    rigid = _as_rigid(raw.get("rigid", {}))
    viewer = _as_viewer(raw.get("viewer", {}))

    return ProjectConfig(simulation=simulation, container=container, rigid=rigid, viewer=viewer)
