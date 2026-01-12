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
├─ configs/                     # Runtime presets (particle count, solver params)
├─ docs/                        # Design notes, derivations, troubleshooting
│  └─ project_overview.md
├─ src/
│  ├─ app/                      # Entry points (viewers)
│  │  ├─ realtime_viewer.py     # Raw particle viewer
│  │  └─ smoothed_viewer.py     # A more realistic viewer with surface reconstruction
│  ├─ kernels/
│  │  └─ fluids.py              # Warp kernels for PBF/SPH pipeline (density, lambda, delta, xsph, etc.)
│  ├─ sim/
│  │  ├─ particles.py           # Particle buffers, initialization
│  │  ├─ grid.py                # Hash-grid logic
│  │  ├─ solver.py              # Frame loop orchestration
│  │  ├─ pressure.py            # Host-side wrappers for PBF constraint solve
│  │  ├─ collision.py           # Container procedural SDF & collision handling
│  │  └─ rigid_spheres.py       # Rigid sphere dynamics & coupling
│  └─ utils/
│     └─ config.py              # Configuration loading
├─ .gitignore
├─ README.md
└─ requirements.txt
```

### Key modules and responsibilities

- `sim/particles.py`: defines the struct-of-arrays Warp buffers (`positions`, `velocities`, `velocities_temp`, `prev_positions`, `densities`, `lambdas`, etc.) and handles device/host synchronization.
- `sim/grid.py`: builds the uniform hash grid and exposes neighbor iteration helpers.
- `sim/pressure.py`: orchestrates the iterative PBF solve by invoking kernels from `kernels/fluids.py`.
- `sim/collision.py`: implements procedural box SDF (`signed_distance_box`) and applied container collisions (`container_collision_kernel`).
- `sim/rigid_spheres.py`: manages two translation-only rigid spheres, computes water–sphere impulses, and handles sphere–container / sphere–sphere collisions.
- `sim/solver.py`: high-level simulation stepper that runs prediction, hash grid build, PBF iterations, velocity update (with XSPH), sphere coupling, and collisions in order.
- `app/realtime_viewer.py`: particle-only viewer for quick inspection and debugging.
- `app/smoothed_viewer.py`: smoothed surface viewer that also renders and couples the two rigid spheres.

## Architecture overview

| Component           | Role                                                                                             |
| ------------------- | ------------------------------------------------------------------------------------------------ |
| Fluid particles     | Drive dynamics; each particle stores `x`, `v`, `density`, `lambda`, `delta_x`          |
| Rigid spheres       | Two translation-only balls storing center, velocity, radius, inverse mass; moved by contact and gravity |
| Neighbor grid       | Uniform hash grid rebuilt every substep for O(N) neighbor queries                                |
| Pressure solver     | PBF constraint projection enforcing incompressibility                                      |
| Integration         | Semi-implicit Euler for prediction plus velocity update from corrected positions                 |
| Fluid–rigid contact | PBD-style contact constraints for water–sphere, sphere–container, and sphere–sphere interactions |
| Container collision | Procedural SDF evaluated in container-local space; push-out and velocity damping                 |
| Runtime loop        | Python orchestrator backed by PyVista (VTK) for visualization/input, keeping kernels on GPU      |

## State layout (Warp arrays)

- `positions`, `velocities`, `velocities_temp`, `prev_positions`: `wp.array(dtype=wp.vec3)`
- `densities`, `lambdas`: `wp.array(dtype=float)`
- `delta_pos`: `wp.array(dtype=wp.vec3)` used for accumulated correction per iteration
- Rigid spheres: `centers`, `velocities`: `wp.array(dtype=wp.vec3)`; `radii`, `inv_masses`, and impulse accumulators: `wp.array(dtype=float)`
- Grid buffers: `cell_keys`, `sorted_indices`, `cell_offsets`: `wp.array(int)`
- Container data: `half_extents` (wp.vec3) and `translation` (wp.vec3) for procedural SDF

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

## Pressure solver details (PBF style)

- Constraint: `C_i = rho_i / rho0 - 1`. 
- Lambda computation:
  ```text
  lambda_i = -C_i / (Σ_j |∇W_ij|^2 + ε)
  ```
- Position delta:
  ```text
  Δx_i = Σ_j (lambda_i + lambda_j) * ∇W_ij
  ```
- Iterate a fixed number of times (e.g. 6) per substep.

## Viscosity via XSPH blending

- Formula per particle:
  ```text
  v_i = v_i + c_xsph * Σ_j (m_j / ρ_j) * (v_j - v_i) * W_ij
  ```
- Implemented via double buffering (`copy_vec3_array_kernel` then `xsph_kernel`) to avoid race conditions.

## Container handling via Procedural SDF

- The container is an axis-aligned box with purely kinematic translation (no rotation).
- A Warp function `signed_distance_box(p, half_extents)` computes the SDF. 
- In our convention, `dist > 0` means **outside** the fluid domain (i.e. outside the inner box volume), so we push particles back in.
- Gradients are estimated via finite differences to determine the push direction.

## Warp kernel breakdown

- **Fluid Kernels** (`src/kernels/fluids.py`):
  - `predict_positions_kernel`
  - `density_kernel`
  - `lambda_kernel`
  - `delta_kernel`
  - `apply_delta_kernel`
  - `update_velocity_kernel`
  - `copy_vec3_array_kernel`
  - `xsph_kernel`
- **Collision/Coupling Kernels** (`src/sim/collision.py`, `src/sim/rigid_spheres.py`):
  - `container_collision_kernel`
  - `sphere_sphere_collision_kernel`
  - `sphere_water_coupling_kernel`
  - `integrate_spheres_kernel`
  - `sphere_container_collision_kernel`

## Visualization & Interaction (PyVista)

The project uses **PyVista** (VTK-based) for real-time visualization with an interactive update loop.

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
    sim_state.container.set_translation(input_state.translation)
    
    for _ in range(sim_state.substeps):
        # Fluid prediction
        predict_positions(sim_state)
        build_grid(sim_state)
        
        # PBF Iterations
        for _ in range(sim_state.iterations):
            compute_density(sim_state)
            compute_lambdas(sim_state)
            compute_deltas(sim_state)
            apply_deltas(sim_state)
            
        update_velocities(sim_state)
        apply_xsph(sim_state)

        # Rigid & Coupling
        collide_spheres_container(sim_state)
        collide_spheres_mutual(sim_state)
        couple_fluid_spheres(sim_state) # Fluid->Sphere impulses
        integrate_spheres(sim_state)    # Update spheres
        
        # Fluid-Container
        collide_fluid_container(sim_state)
```

## Implementation Completed

1. ✅ Scaffolded `src/` tree with clean module imports
2. ✅ Implemented particle buffer management (`sim/particles.py`) with GPU arrays
3. ✅ Built hash-grid via Warp's `wp.HashGrid` (`sim/grid.py`)
4. ✅ Coded PBF pressure kernels (`kernels/fluids.py`)
5. ✅ Implemented XSPH with normalized weighted averaging
6. ✅ Added procedural SDF box collision (`sim/collision.py`)
7. ✅ Implemented rigid sphere dynamics and two-way coupling (`sim/rigid_spheres.py`)
8. ✅ Connected PyVista viewer (`app/realtime_viewer.py`, `app/smoothed_viewer.py`) with keyboard controls and real-time updates

