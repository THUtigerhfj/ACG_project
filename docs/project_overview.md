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
- ✅ Two translation-only rigid spheres coupled to the fluid via PBD-style contact impulses (one buoyant, one sinking)
- ✅ Sphere–container and sphere–sphere collisions

## Key Features

- **Position-Based Fluids (PBF) incompressibility**  
  Can be seen as am implementation of PBD on fluid. The fluid is advanced with a position-based formulation: after a gravity-based prediction step, particle positions are iteratively corrected so that a density constraint \(C_i = \rho_i/\rho_0 - 1\) is (approximately) zero. This enforces near-incompressibility without explicitly integrating pressure forces, and it is unconditionally stable for reasonably large time steps.

- **XSPH viscosity for smooth, damped motion**  
  After the PBF projection, velocities are updated from corrected positions and then blended with neighbors using an XSPH term. This behaves like a tunable viscosity: it damps relative motion between nearby particles, smoothing out noise and preventing persistent oscillations when the container or rigid bodies stop moving.

- **PBD-style collision and rigid coupling**  
  Collisions against the container and rigid spheres are handled as position-based constraints: we compute signed distances (SDF for the box, analytic for spheres), project particles or sphere centers out of penetration, and adjust normal velocities. For water–sphere interaction, the particle correction and velocity change are converted into equal-and-opposite impulses on the sphere, giving a simple two-way coupling between fluid and rigid bodies without modifying the core PBF density solve.

- **Single integrated pipeline**  
  Each substep runs prediction → PBF density projection → XSPH viscosity on the fluid, then applies PBD-style contacts in this order: sphere–container, sphere–sphere, and fluid–sphere, followed by container–fluid collisions. This keeps all interactions within one consistent, position-based framework while still separating concerns between incompressibility (PBF), smoothing/damping (XSPH), and geometry interaction (PBD-style contacts).

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
- `sim/rigid_spheres.py`: manages two translation-only rigid spheres, computes water–sphere impulses and sphere–container / sphere–sphere collision corrections.
- `sim/solver.py`: high-level simulation stepper that runs gravity prediction, grid build, PBF iterations, velocity update, sphere coupling, and container collision in the required order.
- `app/realtime_viewer.py`: particle-only viewer for quick inspection and debugging.
- `app/smoothed_viewer.py`: smoothed surface viewer that also renders and couples the two rigid spheres.

## Architecture overview

| Component           | Role                                                                                             |
| ------------------- | ------------------------------------------------------------------------------------------------ |
| Fluid particles     | Drive dynamics; each particle stores `x`, `v`, `density`, `lambda`, `delta_x`          |
| Rigid spheres       | Two translation-only balls storing center, velocity, radius, inverse mass; moved by contact and gravity |
| Neighbor grid       | Uniform hash grid rebuilt every substep for O(N) neighbor queries                                |
| Pressure solver     | PBF/DFSPH constraint projection enforcing incompressibility                                      |
| Integration         | Semi-implicit Euler for prediction plus velocity update from corrected positions                 |
| Fluid–rigid contact | PBD-style contact constraints for water–sphere, sphere–container, and sphere–sphere interactions |
| Container collision | SDF evaluated in container space, apply push-out and normal damping                              |
| Runtime loop        | Python orchestrator backed by Warp Viewer for camera/input, keeping kernels on GPU               |

## State layout (Warp arrays)

- `positions`, `velocities`, `positions_prev`: `wp.array(dtype=wp.vec3)`
- `densities`, `lambdas`: `wp.array(dtype=float)`
- `delta_pos`: `wp.array(dtype=wp.vec3)` used for accumulated correction per iteration
- Rigid spheres: `centers`, `velocities`: `wp.array(dtype=wp.vec3)`; `radii`, `inv_masses`, and impulse accumulators: `wp.array(dtype=float)`
- Grid buffers: `cell_keys`, `sorted_indices`, `cell_offsets`, `neighbors` (optional compact list) stored as `wp.array(int)`
- Container data: `sdf_values` (3D texture), `sdf_resolution`, `container_transform` (4x4), and `inv_transform`

## Per-frame pipeline

For each rendered frame run `substeps` times (2–4 typical):

1. **User input**: viewer updates the kinematic container translation from keyboard input.
2. **Apply forces / predict (fluid)**: `v += dt * gravity`, `x_pred = x + dt * v`.
3. **Build grid**: hash `x_pred`, sort, produce `cell_offsets`.
4. **Pressure iterations (fluid)** (4–8 passes):
  - `compute_density()` using smoothing kernel sums.
  - `compute_lambda()` evaluating constraint `C_i = rho_i/rho0 - 1`.
  - `compute_pos_delta()` accumulating pairwise position corrections `Δx_i`.
  - `apply_pos_delta()` updating predicted positions.
5. **Update fluid velocities + viscosity**: `v = (x_corrected - x_prev) / dt`, then apply XSPH blending to introduce controllable viscosity/damping.
6. **Rigid–container contact**: project rigid sphere centers back inside the box and damp outward normal velocity.
7. **Rigid–rigid contact**: resolve sphere–sphere overlap and normal relative velocity using a PBD-style contact constraint.
8. **Fluid–rigid coupling**: project fluid particles that lie inside a sphere back to the surface; convert the resulting change in particle velocity into equal-and-opposite impulses on that sphere and integrate sphere motion with gravity.
9. **Container–fluid collision**: sample container SDF for particles; if `d < 0`, push particles out along gradient and zero outward normal velocity.
10. **Swap buffers**: set `x_prev = x_corrected` for the next substep.

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
- `container_collision_kernel`: sample SDF, push out, damp particle velocity.
- `sphere_container_collision_kernel`: push rigid sphere centers back inside the container and remove outward normal velocity.
- `sphere_sphere_collision_kernel`: resolve overlap and normal relative motion between the two rigid spheres.
- `sphere_water_coupling_kernel`: handle fluid–sphere contact by projecting particles out of the sphere and accumulating equal-and-opposite impulses onto the sphere.
- `integrate_spheres_kernel`: apply accumulated impulses and gravity to rigid spheres and advance their centers.

## Visualization & Interaction (PyVista)

- The project uses **PyVista** (VTK-based) for real-time visualization with an interactive update loop.
- Particles are rendered as blue spheres using point cloud rendering with `render_points_as_spheres=True`.
- Container is displayed as a wireframe box that updates position in real-time.
- The main loop uses `plotter.show(interactive_update=True)` with explicit `plotter.update()` calls for smooth animation.

### Keyboard Controls

| Key | Action                      |
| --- | --------------------------- |
| A/D | Move container along X axis |
| X/S | Move container along Y axis |
| Q/E | Move container along Z axis |
| R   | Reset container to origin   |

### Mouse Controls

| Input      | Action        |
| ---------- | ------------- |
| Left-drag  | Rotate camera |
| Right-drag | Pan camera    |
| Scroll     | Zoom in/out   |

## Minimal pseudo-code

```python
def simulate_frame(sim_state, input_state):
  # Update kinematic container from input
  sim_state.container.set_transform(input_state.mouse_matrix)
    for _ in range(sim_state.substeps):
    # Fluid prediction + incompressibility
    predict_positions(sim_state)          # gravity + x_pred
    build_grid(sim_state)                 # neighbor structure
    for _ in range(sim_state.pressure_iters):
      compute_density(sim_state)        # rho_i from neighbors
      compute_lambdas(sim_state)        # density constraint C_i
      accumulate_position_deltas(sim_state)
      apply_position_deltas(sim_state)  # update x
    update_velocities(sim_state)          # v = (x - x_prev)/dt
    apply_xsph(sim_state)                 # viscosity-like damping

    # Rigid spheres: container / mutual / fluid coupling
    collide_spheres_with_container(sim_state)
    collide_spheres_with_each_other(sim_state)
    couple_fluid_and_spheres(sim_state)   # project particles, accumulate impulses
    integrate_spheres(sim_state)          # apply impulses + gravity

    # Fluid vs container
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

