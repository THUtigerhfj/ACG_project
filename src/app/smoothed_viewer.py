"""Entry point that launches the Warp-based SPH simulator."""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Tuple, TYPE_CHECKING

import warp as wp
import numpy as np

try:  # pragma: no cover - heavy UI dependency, optional for tests
    import pyvista as pv
except Exception:  # pragma: no cover - viewer is optional at test time
    pv = None

PvPlotter = Any
PvPolyData = Any

if TYPE_CHECKING:
    Vec3Array = Any
    ScalarArray = Any
else:
    Vec3Array = wp.array(dtype=wp.vec3)
    ScalarArray = wp.array(dtype=float)

from src.kernels.fluids import kernel_poly6
from src.sim.solver import PBFSimulator, build_simulation_state
from src.utils.config import ProjectConfig, load_config

MOVE_STEP = 0.06   # Container displacement target per key press (world units)
MOVE_DURATION = 0.1  # Simulated seconds a key press keeps the container moving
MOVE_SPEED = MOVE_STEP / MOVE_DURATION  # Derived constant for smooth motion


@wp.kernel
def sample_density_kernel(
    base_points: Vec3Array,
    translation: wp.vec3,
    field_out: ScalarArray,
    positions: Vec3Array,
    particle_mass: float,
    radius: float,
    poly6_coeff: float,
    grid_id: wp.uint64,
):
    """Sample an SPH-style density field over a regular grid via Warp."""

    tid = wp.tid()
    sample_pos = base_points[tid] + translation
    rho = float(0.0)

    query = wp.hash_grid_query(grid_id, sample_pos, radius)
    j = int(0)
    while wp.hash_grid_query_next(query, j):
        r = wp.length(sample_pos - positions[j])
        rho += particle_mass * kernel_poly6(r, radius, poly6_coeff)

    field_out[tid] = rho


@dataclass
class SurfaceMesher:
    """GPU-accelerated water surface reconstruction using Warp."""

    simulator: PBFSimulator
    device: str
    grid_resolution: Tuple[int, int, int] = (64, 48, 64)
    field_margin: float = 0.04
    iso_fraction: float = 0.35
    smooth_iters: int = 30
    smooth_relax: float = 0.08

    _base_points: wp.array = field(init=False, repr=False)
    _field_buffer: wp.array = field(init=False, repr=False)
    _dims: Tuple[int, int, int] = field(init=False)
    _spacing: Tuple[float, float, float] = field(init=False)
    _min_offset: Tuple[float, float, float] = field(init=False)

    def __post_init__(self) -> None:
        container = self.simulator.state.container
        hx, hy, hz = container.half_extents
        margin = self.field_margin

        mins = (-hx - margin, -hy - margin, -hz - margin)
        maxs = (hx + margin, hy + margin, hz + margin)
        nx, ny, nz = self.grid_resolution

        xs = np.linspace(mins[0], maxs[0], nx, dtype=np.float32)
        ys = np.linspace(mins[1], maxs[1], ny, dtype=np.float32)
        zs = np.linspace(mins[2], maxs[2], nz, dtype=np.float32)
        grid = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1).reshape(-1, 3)

        self._base_points = wp.array(grid, dtype=wp.vec3, device=self.device)
        self._field_buffer = wp.zeros(grid.shape[0], dtype=float, device=self.device)
        self._dims = (nx, ny, nz)
        self._spacing = (
            float((maxs[0] - mins[0]) / max(nx - 1, 1)),
            float((maxs[1] - mins[1]) / max(ny - 1, 1)),
            float((maxs[2] - mins[2]) / max(nz - 1, 1)),
        )
        self._min_offset = mins

    def _grid_origin(self, translation: Tuple[float, float, float]) -> Tuple[float, float, float]:
        return (
            translation[0] + self._min_offset[0],
            translation[1] + self._min_offset[1],
            translation[2] + self._min_offset[2],
        )

    def _translation_vec(self, translation: Tuple[float, float, float]) -> wp.vec3:
        return wp.vec3(float(translation[0]), float(translation[1]), float(translation[2]))

    def rebuild_surface(self) -> PvPolyData:
        particles = self.simulator.state.particles
        container = self.simulator.state.container

        # Ensure neighbor structure matches the latest particle positions.
        self.simulator.grid.build(particles.positions)

        wp.launch(
            sample_density_kernel,
            dim=self._base_points.shape[0],
            inputs=[
                self._base_points,
                self._translation_vec(container.translation),
                self._field_buffer,
                particles.positions,
                self.simulator.mass,
                self.simulator.h,
                self.simulator.poly6_coeff,
                self.simulator.grid.id,
            ],
            device=self.device,
        )

        field = self._field_buffer.numpy()
        if not np.isfinite(field).any():
            return pv.PolyData()

        # VTK/PyVista expect point data flattened with X varying fastest.
        # Our `base_points` were created with meshgrid(indexing='ij') and
        # flattened, which makes Z vary fastest in the `field` array. To map
        # the sampled field into VTK's X-fast layout while keeping
        # `dimensions`, `spacing`, and `origin` in world (X,Y,Z) order we
        # reshape and transpose the density field before assigning it.
        nx, ny, nz = self._dims
        sx, sy, sz = self._spacing
        ox, oy, oz = self._grid_origin(container.translation)

        # reshape to (nx, ny, nz) where axes correspond to X, Y, Z
        arr = field.reshape((nx, ny, nz))
        # transpose to (nz, ny, nx) so that when flattened, X becomes the
        # fastest varying axis expected by VTK/ImageData
        reordered = arr.transpose((2, 1, 0)).ravel()

        grid = pv.ImageData()
        grid.dimensions = (nx, ny, nz)
        grid.spacing = (sx, sy, sz)
        grid.origin = (ox, oy, oz)
        grid.point_data["density"] = reordered

        max_rho = float(field.max())
        min_rho = float(field.min())
        if max_rho <= min_rho:
            return pv.PolyData()

        iso = min_rho + (max_rho - min_rho) * self.iso_fraction
        surface = grid.contour([iso], scalars="density")

        if surface.n_points == 0:
            return pv.PolyData()

        return surface.smooth(n_iter=self.smooth_iters, relaxation_factor=self.smooth_relax)


@dataclass
class ViewerRuntime:
    simulator: PBFSimulator
    config: ProjectConfig
    device: str
    max_frames: Optional[int] = None

    # PyVista-specific runtime objects (created lazily in run())
    plotter: Optional[PvPlotter] = field(default=None, init=False)
    particle_cloud: Optional[PvPolyData] = field(default=None, init=False)
    container_outer_mesh: Optional[PvPolyData] = field(default=None, init=False)
    container_edges_mesh: Optional[PvPolyData] = field(default=None, init=False)
    container_inner_edges_mesh: Optional[PvPolyData] = field(default=None, init=False)
    water_mesh: Optional[PvPolyData] = field(default=None, init=False)
    water_actor: Any = field(default=None, init=False)
    sphere_meshes: list[PvPolyData] = field(default_factory=list, init=False)
    sphere_actors: list[Any] = field(default_factory=list, init=False)
    surface_mesher: Optional[SurfaceMesher] = field(default=None, init=False)

    # State
    _frame_count: int = field(default=0, init=False)
    _simple_mode: bool = field(default=False, init=False)
    _last_surface_ms: float = field(default=0.0, init=False)
    _container_velocity: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float32), init=False, repr=False
    )
    _velocity_until: float = field(default=0.0, init=False, repr=False)
    _sim_time: float = field(default=0.0, init=False, repr=False)

    def _build_particle_cloud(self) -> PvPolyData:
        positions = self.simulator.state.particles.positions.numpy().copy()
        return pv.PolyData(positions)

    def _build_container_meshes(self) -> Tuple[PvPolyData, PvPolyData, PvPolyData]:
        container = self.simulator.state.container
        hx, hy, hz = container.half_extents
        cx, cy, cz = container.translation
        thickness = container.wall_thickness

        outer_bounds = (
            cx - (hx + thickness), cx + (hx + thickness),
            cy - (hy + thickness), cy + (hy + thickness),
            cz - (hz + thickness), cz + (hz + thickness)
        )
        inner_bounds = (cx - hx, cx + hx, cy - hy, cy + hy, cz - hz, cz + hz)

        outer_box = pv.Box(bounds=outer_bounds)
        inner_edges = pv.Box(bounds=inner_bounds).extract_feature_edges()
        outer_edges = outer_box.extract_feature_edges()

        return outer_box, outer_edges, inner_edges

    def _update_container_meshes(self) -> None:
        if self.container_outer_mesh is None:
            return
        container = self.simulator.state.container
        hx, hy, hz = container.half_extents
        cx, cy, cz = container.translation
        thickness = container.wall_thickness

        outer_bounds = (
            cx - (hx + thickness), cx + (hx + thickness),
            cy - (hy + thickness), cy + (hy + thickness),
            cz - (hz + thickness), cz + (hz + thickness)
        )
        inner_bounds = (cx - hx, cx + hx, cy - hy, cy + hy, cz - hz, cz + hz)

        new_outer = pv.Box(bounds=outer_bounds)
        new_outer_edges = new_outer.extract_feature_edges()
        new_inner_edges = pv.Box(bounds=inner_bounds).extract_feature_edges()

        self.container_outer_mesh.copy_from(new_outer)
        if self.container_edges_mesh is not None:
            self.container_edges_mesh.copy_from(new_outer_edges)
        if self.container_inner_edges_mesh is not None:
            self.container_inner_edges_mesh.copy_from(new_inner_edges)

    def _build_sphere_meshes(self) -> None:
        spheres = self.simulator.state.spheres
        centers = spheres.centers.numpy()
        radii = spheres.radii.numpy()

        self.sphere_meshes = []
        self.sphere_actors = []
        for i in range(centers.shape[0]):
            mesh = pv.Sphere(center=centers[i], radius=float(radii[i]), theta_resolution=32, phi_resolution=32)
            actor = self.plotter.add_mesh(mesh, color=(0.9, 0.35, 0.2) if i == 0 else (0.2, 0.35, 0.9), opacity=0.8)
            self.sphere_meshes.append(mesh)
            self.sphere_actors.append(actor)

    def _update_sphere_meshes(self) -> None:
        if not self.sphere_meshes:
            return
        spheres = self.simulator.state.spheres
        centers = spheres.centers.numpy()
        radii = spheres.radii.numpy()
        for i, mesh in enumerate(self.sphere_meshes):
            updated = pv.Sphere(center=centers[i], radius=float(radii[i]), theta_resolution=32, phi_resolution=32)
            try:
                mesh.copy_from(updated)
            except Exception:
                # If copy fails, replace actor to stay robust.
                if self.plotter is not None and i < len(self.sphere_actors):
                    self.plotter.remove_actor(self.sphere_actors[i])
                    actor = self.plotter.add_mesh(updated, color=(0.9, 0.35, 0.2) if i == 0 else (0.2, 0.35, 0.9), opacity=0.8)
                    self.sphere_actors[i] = actor
                    self.sphere_meshes[i] = updated

    def _setup_lights(self) -> None:
        key = pv.Light(position=(3, 3, 4), focal_point=(0, 0, 0), intensity=0.9)
        fill = pv.Light(position=(-3, 1, 2), focal_point=(0, 0, 0), intensity=0.45)
        rim = pv.Light(position=(0, -3, 1), focal_point=(0, 0, 0), intensity=0.35)
        self.plotter.add_light(key)
        self.plotter.add_light(fill)
        self.plotter.add_light(rim)

    def _fallback_to_particles(self, reason: str = "") -> None:
        if self._simple_mode:
            return
        print(f"Falling back to particle mode: {reason}")
        self._simple_mode = True
        self.particle_cloud = self._build_particle_cloud()
        self.water_mesh = self.particle_cloud
        self.water_actor = self.plotter.add_mesh(
            self.particle_cloud,
            color="deepskyblue",
            point_size=8,
            render_points_as_spheres=True,
            opacity=0.9,
            name="water_particles",
        )

    def _update_water_representation(self) -> None:
        if self._simple_mode or self.surface_mesher is None:
            positions = self.simulator.state.particles.positions.numpy()
            if self.particle_cloud is not None:
                self.particle_cloud.points = positions
            return

        start = time.perf_counter()
        surface = self.surface_mesher.rebuild_surface()
        self._last_surface_ms = (time.perf_counter() - start) * 1000.0

        if surface is None or surface.n_points == 0:
            self._fallback_to_particles(reason="Surface rebuild empty")
            return

        if self.water_mesh is None:
            self.water_mesh = surface
            self.water_actor = self.plotter.add_mesh(
                surface,
                color=(0.15, 0.4, 0.9),
                opacity=0.72,
                smooth_shading=True,
                specular=0.8,
                specular_power=80,
                diffuse=0.5,
                ambient=0.2,
                name="water_surface",
            )
        else:
            try:
                self.water_mesh.copy_from(surface)
            except Exception:
                self._fallback_to_particles(reason="Mesh copy failed")

    def _schedule_container_move(self, direction: Tuple[float, float, float]) -> None:
        """Convert a key press into a short-lived velocity impulse."""
        if not any(direction):
            return
        self._container_velocity[:] = np.asarray(direction, dtype=np.float32) * MOVE_SPEED
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
        container.set_translation(
            (
                tx + float(vx * dt),
                ty + float(vy * dt),
                tz + float(vz * dt),
            )
        )

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

        self.plotter = pv.Plotter(window_size=(1280, 720), lighting="none")
        # Remove PyVista default key bindings (e.g., 'w' toggles wireframe) to avoid
        # conflicts with our movement controls.
        try:
            self.plotter.clear_events()
        except Exception:
            pass

        bg_color = self.config.viewer.background_color
        if isinstance(bg_color, (list, tuple)):
            bg_color = tuple(bg_color)
        self.plotter.set_background(bg_color)

        # Lighting tuned for glass/water
        self._setup_lights()

        # Build container geometry
        self.container_outer_mesh, self.container_edges_mesh, self.container_inner_edges_mesh = self._build_container_meshes()

        self.plotter.add_mesh(
            self.container_outer_mesh,
            color=(0.85, 0.92, 0.97),
            opacity=0.2,
            smooth_shading=True,
            specular=0.9,
            specular_power=100,
            ambient=0.15,
            diffuse=0.4,
            name="container_glass",
        )

        self.plotter.add_mesh(
            self.container_edges_mesh,
            color="white",
            line_width=3,
            opacity=0.85,
            name="container_edges",
        )

        self.plotter.add_mesh(
            self.container_inner_edges_mesh,
            color=(0.7, 0.85, 0.95),
            line_width=1.5,
            opacity=0.5,
            name="container_inner_edges",
        )

        # Build spheres
        self._build_sphere_meshes()

        # Initialize mesher and first surface
        self.surface_mesher = SurfaceMesher(self.simulator, self.device)
        self._update_water_representation()

        # Set up camera: use top-down view so gravity (-Z) maps to screen downwards.
        # When using an oblique camera like (1,1,1) the -Z axis can project into
        # screen depth which makes falling look like it's toward the "back".
        # Place the camera above the scene and look at the origin.
        self.plotter.camera_position =  [(0, 4, 0), (0, 0, 0), (0, 0, 1)]
        self.plotter.camera.zoom(1.2)

        # Add instructions text (note: 'X' is used instead of PyVista's default 'w')
        self.plotter.add_text(
            "A/D: Move X | X/S: Move Y | Q/E: Move Z | R: Reset\n"
            "Mouse: Rotate/Pan/Zoom",
            position="lower_left",
            font_size=10,
            color="white",
        )

        # Add a small axes widget so world X/Y/Z directions are visible
        try:
            self.plotter.add_axes()
        except Exception:
            pass

        def schedule_move(direction: Tuple[float, float, float]):
            self._schedule_container_move(direction)

        def reset_container():
            self.simulator.state.container.set_translation((0.0, 0.0, 0.0))
            self._container_velocity[:] = 0.0
            self._velocity_until = 0.0
            print("Container reset to origin")

        # Register key events with smooth motion scheduling
        self.plotter.add_key_event("a", lambda: schedule_move((-1.0, 0.0, 0.0)))
        self.plotter.add_key_event("d", lambda: schedule_move((1.0, 0.0, 0.0)))
        self.plotter.add_key_event("s", lambda: schedule_move((0.0, -1.0, 0.0)))
        # Use 'x' for +Y to avoid PyVista default wireframe toggle on 'w' and keep close to 's'
        self.plotter.add_key_event("x", lambda: schedule_move((0.0, 1.0, 0.0)))
        self.plotter.add_key_event("q", lambda: schedule_move((0.0, 0.0, -1.0)))
        self.plotter.add_key_event("e", lambda: schedule_move((0.0, 0.0, 1.0)))
        self.plotter.add_key_event("r", reset_container)

        # Show window in interactive mode (non-blocking)
        self.plotter.show(interactive_update=True, auto_close=False)

        print("Starting simulation loop...")
        print("Controls: A/D (X), W/S (Y), Q/E (Z), R (reset)")
        
        target_dt = 1.0 / 60.0  # Target 60 FPS for rendering
        sim_step_dt = float(self.simulator.dt * self.simulator.substeps) # Time for one simulation.step()
        self._sim_time = 0.0
        
        try:
            while True:
                frame_start = time.perf_counter()

                # Smooth container motion occurs using simulated time deltas
                self._apply_container_motion(sim_step_dt)

                # Run simulation step
                self.simulator.step()
                self._frame_count += 1
                self._sim_time += sim_step_dt

                # Update water representation
                self._update_water_representation()

                # Update spheres
                self._update_sphere_meshes()

                # Update container mesh
                self._update_container_meshes()

                # Render
                self.plotter.update()

                # Debug output
                if self._frame_count % 60 == 0:
                    mode = "Particles" if self._simple_mode else "Surface"
                    print(f"Frame {self._frame_count}: mode={mode}")

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