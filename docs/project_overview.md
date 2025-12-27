# Real-Time SPH Simulator Overview

## Project Goals

- Real-time incompressible SPH water contained in a rectangular box the user controls with keyboard inputs (container is purely kinematic).
- Implement every hot loop in Python + NVIDIA Warp for GPU throughput while keeping the rest of the app in idiomatic Python.
- Use a signed-distance field (SDF) to describe container walls; no force feedback from fluid to container is required.
- Ship an interactive playground with PyVista-based visualization and keyboard controls for container movement.

## Current Implementation Status

- ✅ Position-Based Fluids (PBF) pressure solver
- ✅ XSPH viscosity with normalized weighted averaging
- ✅ Hash grid neighbor search via Warp
- ✅ SDF-based container collision
- ✅ Real-time PyVista visualization (~60 FPS)
- ✅ Keyboard controls for container movement (A/D/W/S/Q/E/R)

## Updated repository layout

```text
ACG_project/
├─ docs/                        # Design notes, derivations, troubleshooting
│  └─ project_overview.md
├─ configs/                     # Runtime presets (particle count, solver params)
├─ src/
│  ├─ app/                      # Entry points (viewer, CLI) and mouse input plumbing
│  │  └─ realtime_viewer.py     # Wraps NVIDIA Warp Viewer (OpenGL/ImGui) for live control
│  ├─ sim/
│  │  ├─ particles.py          # Particle buffers, initialization utilities
│  │  ├─ grid.py               # Hash-grid build & neighborhood queries
│  │  ├─ solver.py             # Frame loop orchestration & substep management
│  │  ├─ pressure.py           # PBF/DFSPH constraint evaluation kernels
│  │  └─ collision.py          # Container SDF sampling & response
│  ├─ kernels/                 # Warp kernels grouped by stage (density, lambda, etc.)
│  ├─ utils/                   # Math helpers, profiling, parameter validation
│  └─ viz/                     # Lightweight scatter/isosurface previews (optional)
├─ assets/
│  ├─ initial_states/          # Particle lattices or cached npz setups
│  └─ sdf/                     # Binary SDF grids for alternate container shapes
├─ scripts/                    # Dev utilities (profilers, cache dumpers)
├─ tests/                      # Unit tests for kernels and integrators
└─ README.md                   # Quick start guide
```

### Key modules and responsibilities

- `sim/particles.py`: defines the struct-of-arrays Warp buffers (`positions`, `velocities`, `densities`, `lambdas`, etc.) and handles device/host synchronization when needed.
- `sim/grid.py`: builds the uniform hash grid (compute keys, radix sort, prefix offsets) and exposes neighbor iteration helpers usable inside Warp kernels.
- `sim/pressure.py`: implements Position-Based Fluids style lambda solve plus position deltas; holds iteration counts and convergence checks.
- `sim/collision.py`: stores container SDF data/transform, samples signed distance, returns push-out vectors, and applies simple velocity damping along the collision normal.
- `sim/solver.py`: high-level simulation stepper that runs gravity prediction, grid build, PBF iterations, velocity update, and collision in the required order.
- `app/realtime_viewer.py`: uses the Warp Viewer (OpenGL + ImGui) to provide mouse-driven gizmos, draw particles, and send translation updates into the simulation loop.

## Architecture overview

| Component | Role |
| --- | --- |
| Fluid particles | Drive dynamics; each particle stores `x`, `v`, `density`, `lambda`, `delta_x` |
| Neighbor grid | Uniform hash grid rebuilt every substep for O(N) neighbor queries |
| Pressure solver | PBF/DFSPH constraint projection enforcing incompressibility |
| Integration | Semi-implicit Euler for prediction plus velocity update from corrected positions |
| Container collision | SDF evaluated in container space, apply push-out and normal damping |
| Runtime loop | Python orchestrator backed by Warp Viewer for camera/input, keeping kernels on GPU |

## State layout (Warp arrays)

- `positions`, `velocities`, `positions_prev`: `wp.array(dtype=wp.vec3)`
- `densities`, `lambdas`: `wp.array(dtype=float)`
- `delta_pos`: `wp.array(dtype=wp.vec3)` used for accumulated correction per iteration
- Grid buffers: `cell_keys`, `sorted_indices`, `cell_offsets`, `neighbors` (optional compact list) stored as `wp.array(int)`
- Container data: `sdf_values` (3D texture), `sdf_resolution`, `container_transform` (4x4), and `inv_transform`

## Per-frame pipeline

For each rendered frame run `substeps` times (2–4 typical):

1. **User input**: Warp Viewer exposes a translation gizmo; update the container transform (translation-only) from current mouse drag.
2. **Apply forces / predict**: `v += dt * gravity`, `x_pred = x + dt * v`.
3. **Build grid**: hash `x_pred`, sort, produce `cell_offsets`.
4. **Pressure iterations** (4–8 passes):
   - `compute_density()` using smoothing kernel sums.
   - `compute_lambda()` evaluating constraint `C_i = rho_i/rho0 - 1`.
   - `compute_pos_delta()` accumulating pairwise position corrections `Δx_i`.
   - `apply_pos_delta()` updating predicted positions.
5. **Update velocities**: `v = (x_corrected - x_prev) / dt`, then apply XSPH blending to introduce controllable viscosity/damping.
6. **Container collisions**: sample SDF; if `d < 0`, push particle out along gradient and zero normal velocity component.
7. **Swap buffers**: set `x_prev = x_corrected` for the next substep.

## Pressure solver details (PBF/DFSPH style)

- Constraint: `C_i = rho_i / rho0 - 1`. We only solve when `C_i > 0` to avoid over-expanding sparse regions.
- Lambda computation:

  ```text
  lambda_i = -C_i / (Σ_j |∇W_ij|^2 + ε)
  ```

  where `∇W_ij` is evaluated with the spiky kernel and `ε ≈ 1e-6` for stability.
- Position delta kernel:

  ```text
  Δx_i = Σ_j (lambda_i + lambda_j) * ∇W_ij
  Δx_i += relaxation * n_i    # optional for boundary thickness
  ```

- Iterate 4–8 times per substep or until `max|C_i| < tol`. Warp makes multiple launches inexpensive when buffers stay on GPU, keeping the fluid nearly incompressible in real time.

## Viscosity via XSPH blending

- After the PBF position corrections, apply an XSPH velocity update to introduce an intuitive “thickness” without solving an additional viscosity PDE.
- Formula per particle:

  ```text
  v_i = v_i + c_xsph * Σ_j (m_j / ρ_j) * (v_j - v_i) * W_ij
  ```

  where `c_xsph` is a dimensionless damping knob (e.g., 0.05–0.2). This term damps relative motion, preventing perpetual oscillations when the container stops moving.
- Implementation detail: reuse the neighbor list built for the PBF solve so the extra kernel is O(N). Warp makes it easy to launch a dedicated `xsph_kernel` right after `update_velocity_kernel`.

## Container handling via SDF

- Represent the axis-aligned box as a voxel SDF in `assets/sdf/box.npz`. Because the container only translates (no rotation/scale), the SDF stays aligned with world axes; we simply offset sample positions by the translation vector instead of recomputing gradients.
- During runtime, transform particle positions into container-local space with the inverse mouse-driven matrix, sample trilinearly, and compute gradients using finite differences.
- Collision kernel logic:

  ```python
  d = sample_sdf(x_local)
  if d < 0:
      n = normalize(grad_sdf(x_local))
      x += (-d + padding) * n        # push outside wall
      v -= wp.dot(v, n) * n          # kill normal component; keep tangential slide
  ```

- Because the container is purely kinematic, no forces are sent back; only particle state changes.

## Warp kernel breakdown

- `build_grid_kernel`: compute cell keys from predicted positions.
- `density_kernel`: iterate over neighboring cells, sum poly6 contributions.
- `lambda_kernel`: reuse neighbors, compute constraint denominator and lambda value.
- `delta_kernel`: apply `(λ_i + λ_j)` and accumulate corrections atomically or via shared memory if necessary.
- `apply_delta_kernel`: add corrections to predicted positions and zero `delta_pos` for next iteration.
- `update_velocity_kernel`: compute `(x - x_prev)/dt`.
- `xsph_kernel`: perform velocity blending to emulate viscosity and damp residual jitter.
- `collision_kernel`: sample SDF, push out, damp velocity.

## Visualization & Interaction (PyVista)

- The project uses **PyVista** (VTK-based) for real-time visualization with an interactive update loop.
- Particles are rendered as blue spheres using point cloud rendering with `render_points_as_spheres=True`.
- Container is displayed as a wireframe box that updates position in real-time.
- The main loop uses `plotter.show(interactive_update=True)` with explicit `plotter.update()` calls for smooth animation.

### Keyboard Controls

| Key | Action |
|-----|--------|
| A/D | Move container along X axis |
| X/S | Move container along Y axis |
| Q/E | Move container along Z axis |
| R   | Reset container to origin |

### Mouse Controls

| Input | Action |
|-------|--------|
| Left-drag | Rotate camera |
| Right-drag | Pan camera |
| Scroll | Zoom in/out |

## Minimal pseudo-code

```python
def simulate_frame(sim_state, input_state):
    sim_state.container.set_transform(input_state.mouse_matrix)
    for _ in range(sim_state.substeps):
        predict_positions(sim_state)
        build_grid(sim_state)
        for _ in range(sim_state.pressure_iters):
            compute_density(sim_state)
            compute_lambdas(sim_state)
            accumulate_position_deltas(sim_state)
            apply_position_deltas(sim_state)
        update_velocities(sim_state)
        resolve_container_collisions(sim_state)
```

## Implementation Completed

1. ✅ Scaffolded `src/` tree with clean module imports
2. ✅ Implemented particle buffer management (`sim/particles.py`) with GPU arrays
3. ✅ Built hash-grid via Warp's `wp.HashGrid` (`sim/grid.py`)
4. ✅ Coded PBF pressure kernels with density clamping for boundary stability
5. ✅ Implemented XSPH with normalized weighted averaging (avoids density division issues)
6. ✅ Added SDF box collision with proper inside/outside detection
7. ✅ Connected PyVista viewer with keyboard controls and real-time updates

## Potential Future Enhancements

- Ghost/boundary particles for improved density estimation at walls
- Surface reconstruction for mesh-based rendering
- Multiple fluid phases with different densities
- Export to USD or Alembic for offline rendering
- ImGui-based parameter tuning panel
