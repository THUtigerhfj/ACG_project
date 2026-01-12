"""Entry point that launches the Warp-based SPH simulator."""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Any

import warp as wp
import numpy as np

try:  # pragma: no cover - heavy UI dependency, optional for tests
    import pyvista as pv
except Exception:  # pragma: no cover - viewer is optional at test time
    pv = None

from src.sim.solver import PBFSimulator, build_simulation_state
from src.utils.config import ProjectConfig, load_config


MOVE_STEP = 0.06  # Target displacement per key press (world units)
MOVE_DURATION = 0.1  # Simulated seconds to apply that displacement
MOVE_SPEED = MOVE_STEP / MOVE_DURATION


@dataclass
class ViewerRuntime:
    simulator: PBFSimulator
    config: ProjectConfig
    device: str
    max_frames: Optional[int] = None

    # PyVista-specific runtime objects (created lazily in run())
    plotter: Optional["pv.Plotter"] = field(default=None, init=False)
    particle_cloud: Optional["pv.PolyData"] = field(default=None, init=False)
    container_mesh: Optional["pv.PolyData"] = field(default=None, init=False)
    sphere_meshes: list["pv.PolyData"] = field(default_factory=list, init=False)
    sphere_actors: list[Any] = field(default_factory=list, init=False)

    # State
    _frame_count: int = field(default=0, init=False)
    _sim_time: float = field(default=0.0, init=False)
    _container_velocity: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float32), init=False, repr=False
    )
    _velocity_until: float = field(default=0.0, init=False, repr=False)

    def _build_initial_particle_cloud(self) -> "pv.PolyData":
        """Create a PyVista point cloud from current particle positions."""
        particles = self.simulator.state.particles
        host = particles.positions.numpy().copy()
        cloud = pv.PolyData(host)
        return cloud

    def _build_container_mesh(self) -> "pv.PolyData":
        """Create a transparent box representing the container."""
        container = self.simulator.state.container
        hx, hy, hz = container.half_extents
        cx, cy, cz = container.translation
        bounds = (cx - hx, cx + hx, cy - hy, cy + hy, cz - hz, cz + hz)
        return pv.Box(bounds=bounds)

    def _update_container_mesh(self) -> None:
        """Update container mesh position."""
        if self.container_mesh is None:
            return
        container = self.simulator.state.container
        hx, hy, hz = container.half_extents
        cx, cy, cz = container.translation
        bounds = (cx - hx, cx + hx, cy - hy, cy + hy, cz - hz, cz + hz)
        new_box = pv.Box(bounds=bounds)
        self.container_mesh.copy_from(new_box)

    def _build_sphere_wireframes(self) -> None:
        """Create simple wireframe spheres to visualize rigid SDF radii."""
        if pv is None:
            return
        spheres = getattr(self.simulator.state, "spheres", None)
        if spheres is None:
            return

        centers = spheres.centers.numpy()
        radii = spheres.radii.numpy()
        self.sphere_meshes = []
        self.sphere_actors = []
        for i in range(centers.shape[0]):
            mesh = pv.Sphere(center=centers[i], radius=float(radii[i]), theta_resolution=24, phi_resolution=24)
            actor = self.plotter.add_mesh(mesh, style="wireframe", color="white", opacity=0.5)
            self.sphere_meshes.append(mesh)
            self.sphere_actors.append(actor)

    def _update_sphere_wireframes(self) -> None:
        if not self.sphere_meshes:
            return
        spheres = getattr(self.simulator.state, "spheres", None)
        if spheres is None:
            return
        centers = spheres.centers.numpy()
        radii = spheres.radii.numpy()
        for i, mesh in enumerate(self.sphere_meshes):
            updated = pv.Sphere(center=centers[i], radius=float(radii[i]), theta_resolution=24, phi_resolution=24)
            try:
                mesh.copy_from(updated)
            except Exception:
                if self.plotter is not None and i < len(self.sphere_actors):
                    self.plotter.remove_actor(self.sphere_actors[i])
                    actor = self.plotter.add_mesh(updated, style="wireframe", color="white", opacity=0.5)
                    self.sphere_actors[i] = actor
                    self.sphere_meshes[i] = updated

    def _schedule_container_move(self, direction: tuple[float, float, float]) -> None:
        """Convert a key press into a short-lived velocity impulse."""
        if not any(direction):
            return
        self._container_velocity[:] = np.asarray(direction, dtype=np.float32) * MOVE_SPEED
        # Use simulation time to expire the motion
        self._velocity_until = self._sim_time + MOVE_DURATION

    def _apply_container_motion(self, dt: float) -> None:
        if dt <= 0.0:
            return
        if not np.any(self._container_velocity):
            return
        if self._sim_time >= self._velocity_until:
            self._container_velocity[:] = 0.0
            return

        container = self.simulator.state.container
        tx, ty, tz = container.translation
        vx, vy, vz = self._container_velocity
        container.set_translation((tx + float(vx * dt), ty + float(vy * dt), tz + float(vz * dt)))

    def run(self) -> None:
        """Launch the PyVista-based interactive viewer.
        
        Controls:
        - A/D: Move container left/right (X axis)
        - W/S: Move container up/down (Y axis)
        - Q/E: Move container forward/backward (Z axis)
        - R: Reset container position
        - Mouse: Rotate/Pan/Zoom camera
        """
        if pv is None:
            raise RuntimeError(
                "PyVista is not available. Install it with 'pip install pyvista'."
            )

        # Build visual geometry
        self.particle_cloud = self._build_initial_particle_cloud()
        self.container_mesh = self._build_container_mesh()

        self.plotter = pv.Plotter(window_size=(1280, 720))

        # Add particle cloud
        self.plotter.add_mesh(
            self.particle_cloud,
            color="deepskyblue",
            point_size=8,
            render_points_as_spheres=True,
        )

        # Add container wireframe
        self.plotter.add_mesh(
            self.container_mesh,
            color="white",
            opacity=0.3,
            style="wireframe",
            line_width=2,
        )

        # Set up camera to match smoothed_viewer (top-down with Z up)
        self.plotter.camera_position = [(0, 4, 0), (0, 0, 0), (0, 0, 1)]
        self.plotter.camera.zoom(1.2)

        # Add instructions text
        self.plotter.add_text(
            "A/D: Move X | W/S: Move Y | Q/E: Move Z | R: Reset\n"
            "Mouse: Rotate/Pan/Zoom",
            position="lower_left",
            font_size=10,
            color="white",
        )

        # Build spheres wireframe visualization (keeps raw particles untouched)
        self._build_sphere_wireframes()

        def reset_container():
            self.simulator.state.container.set_translation((0.0, 0.0, 0.0))
            self._container_velocity[:] = 0.0
            self._velocity_until = 0.0
            print("Container reset to origin")

        # Register key events
        self.plotter.add_key_event("a", lambda: self._schedule_container_move((-1.0, 0.0, 0.0)))
        self.plotter.add_key_event("d", lambda: self._schedule_container_move((1.0, 0.0, 0.0)))
        self.plotter.add_key_event("s", lambda: self._schedule_container_move((0.0, -1.0, 0.0)))
        self.plotter.add_key_event("w", lambda: self._schedule_container_move((0.0, 1.0, 0.0)))
        self.plotter.add_key_event("q", lambda: self._schedule_container_move((0.0, 0.0, -1.0)))
        self.plotter.add_key_event("e", lambda: self._schedule_container_move((0.0, 0.0, 1.0)))
        self.plotter.add_key_event("r", reset_container)

        # Show window in interactive mode (non-blocking)
        self.plotter.show(interactive_update=True, auto_close=False)

        print("Starting simulation loop...")
        print("Controls: A/D (X), W/S (Y), Q/E (Z), R (reset)")
        
        target_dt = 1.0 / 60.0  # Target 60 FPS for rendering
        sim_step_dt = float(self.simulator.dt * self.simulator.substeps)
        self._sim_time = 0.0
        
        try:
            while True:
                frame_start = time.perf_counter()
                
                # Smooth container motion using simulation timestep
                self._apply_container_motion(sim_step_dt)

                # Run simulation step
                self.simulator.step()
                self._frame_count += 1
                self._sim_time += sim_step_dt

                # Update particle positions
                new_positions = self.simulator.state.particles.positions.numpy()
                self.particle_cloud.points = new_positions

                # Update container mesh
                self._update_container_mesh()

                # Update sphere wireframes
                self._update_sphere_wireframes()

                # Render
                self.plotter.update()

                # Debug output
                if self._frame_count % 60 == 0:
                    pos = new_positions[0]
                    print(f"Frame {self._frame_count}: particle[0] = [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")

                # Frame limiting
                if self.max_frames and self._frame_count >= self.max_frames:
                    break

                # Maintain target frame rate
                elapsed = time.perf_counter() - frame_start
                sleep_time = target_dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

                # Check if window closed
                if not self.plotter.window_size:
                    break

        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        self.plotter.close()
        print(f"Simulation ended after {self._frame_count} frames")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Warp SPH demo")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config")
    parser.add_argument("--device", default="cuda", help="Target device (cuda|cpu)")
    parser.add_argument("--frames", type=int, default=None, help="Optional frame limit")
    return parser.parse_args(argv)


def build_simulator(cfg: ProjectConfig, device: str) -> PBFSimulator:
    if not wp.is_device_available(device):
        raise RuntimeError(f"Requested device '{device}' is not available.")

    state = build_simulation_state(cfg, device)
    return PBFSimulator(cfg, state, device=device)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv or sys.argv[1:])
    cfg = load_config(Path(args.config))
    simulator = build_simulator(cfg, args.device)
    ViewerRuntime(simulator, cfg, args.device, max_frames=args.frames).run()


if __name__ == "__main__":
    main()