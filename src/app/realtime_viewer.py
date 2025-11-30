"""Entry point that launches the Warp-based SPH simulator."""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import warp as wp
import numpy as np

try:  # pragma: no cover - heavy UI dependency, optional for tests
    import pyvista as pv
except Exception:  # pragma: no cover - viewer is optional at test time
    pv = None

from src.sim.solver import PBFSimulator, build_simulation_state
from src.utils.config import ProjectConfig, load_config


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

    # State
    _frame_count: int = field(default=0, init=False)

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

        # Set up camera
        self.plotter.camera_position = "iso"
        self.plotter.camera.zoom(1.2)

        # Add instructions text
        self.plotter.add_text(
            "A/D: Move X | W/S: Move Y | Q/E: Move Z | R: Reset\n"
            "Mouse: Rotate/Pan/Zoom",
            position="lower_left",
            font_size=10,
            color="white",
        )

        # Container movement - each key press moves by this amount
        move_step = 0.05

        def move_container(dx, dy, dz):
            container = self.simulator.state.container
            tx, ty, tz = container.translation
            container.set_translation((tx + dx, ty + dy, tz + dz))
            print(f"Container: ({tx+dx:.2f}, {ty+dy:.2f}, {tz+dz:.2f})")

        def reset_container():
            self.simulator.state.container.set_translation((0.0, 0.0, 0.0))
            print("Container reset to origin")

        # Register key events
        self.plotter.add_key_event("a", lambda: move_container(-move_step, 0, 0))
        self.plotter.add_key_event("d", lambda: move_container(move_step, 0, 0))
        self.plotter.add_key_event("s", lambda: move_container(0, -move_step, 0))
        self.plotter.add_key_event("w", lambda: move_container(0, move_step, 0))
        self.plotter.add_key_event("q", lambda: move_container(0, 0, -move_step))
        self.plotter.add_key_event("e", lambda: move_container(0, 0, move_step))
        self.plotter.add_key_event("r", reset_container)

        # Show window in interactive mode (non-blocking)
        self.plotter.show(interactive_update=True, auto_close=False)

        print("Starting simulation loop...")
        print("Controls: A/D (X), W/S (Y), Q/E (Z), R (reset)")
        
        target_dt = 1.0 / 60.0  # Target 60 FPS for rendering
        
        try:
            while True:
                frame_start = time.perf_counter()
                
                # Run simulation step
                self.simulator.step()
                self._frame_count += 1

                # Update particle positions
                new_positions = self.simulator.state.particles.positions.numpy()
                self.particle_cloud.points = new_positions

                # Update container mesh
                self._update_container_mesh()

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