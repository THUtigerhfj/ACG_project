"""Rigid (translation-only) spheres and two-way coupling kernels.

We model spheres as frictionless, non-rotating rigid bodies (center + linear velocity).
Two-way coupling is implemented by projecting fluid particles out of the sphere and
accumulating an equal-and-opposite impulse onto the sphere.

Stability follows common PBD practice:
- contact offset ("slop")
- maximum depenetration per substep (bounded push-out)
- maximum impulse per contact
"""

from __future__ import annotations

from dataclasses import dataclass

import math
import warp as wp


NUM_SPHERES = 2


@dataclass
class RigidSphereState:
    centers: wp.array  # wp.vec3, shape (NUM_SPHERES,)
    velocities: wp.array  # wp.vec3, shape (NUM_SPHERES,)
    radii: wp.array  # float, shape (NUM_SPHERES,)
    inv_masses: wp.array  # float, shape (NUM_SPHERES,)

    impulse_x: wp.array  # float, shape (NUM_SPHERES,)
    impulse_y: wp.array  # float, shape (NUM_SPHERES,)
    impulse_z: wp.array  # float, shape (NUM_SPHERES,)


def sphere_mass_from_density(density: float, radius: float) -> float:
    volume = (4.0 / 3.0) * math.pi * (radius**3)
    return float(density) * float(volume)


def build_rigid_spheres(
    *,
    centers: list[list[float]],
    velocities: list[list[float]],
    radii: list[float],
    densities: list[float],
    device: str,
) -> RigidSphereState:
    if len(centers) != NUM_SPHERES or len(velocities) != NUM_SPHERES or len(radii) != NUM_SPHERES or len(densities) != NUM_SPHERES:
        raise ValueError(f"Expected exactly {NUM_SPHERES} spheres")

    masses = [sphere_mass_from_density(densities[i], radii[i]) for i in range(NUM_SPHERES)]
    inv_masses = [0.0 if m <= 0.0 else 1.0 / m for m in masses]

    centers_arr = wp.array([wp.vec3(*map(float, c)) for c in centers], dtype=wp.vec3, device=device)
    velocities_arr = wp.array([wp.vec3(*map(float, v)) for v in velocities], dtype=wp.vec3, device=device)
    radii_arr = wp.array([float(r) for r in radii], dtype=float, device=device)
    inv_masses_arr = wp.array([float(im) for im in inv_masses], dtype=float, device=device)

    impulse_x = wp.zeros(NUM_SPHERES, dtype=float, device=device)
    impulse_y = wp.zeros(NUM_SPHERES, dtype=float, device=device)
    impulse_z = wp.zeros(NUM_SPHERES, dtype=float, device=device)

    return RigidSphereState(
        centers=centers_arr,
        velocities=velocities_arr,
        radii=radii_arr,
        inv_masses=inv_masses_arr,
        impulse_x=impulse_x,
        impulse_y=impulse_y,
        impulse_z=impulse_z,
    )


@wp.func
def safe_normal(v: wp.vec3, eps: float = 1.0e-8) -> wp.vec3:
    length = wp.length(v)
    if length > eps:
        return v / length
    return wp.vec3(1.0, 0.0, 0.0)


@wp.func
def sdf_box_local(p: wp.vec3, he: wp.vec3) -> float:
    """Signed distance to an axis-aligned box centered at origin with half extents he."""
    q = wp.vec3(wp.abs(p[0]) - he[0], wp.abs(p[1]) - he[1], wp.abs(p[2]) - he[2])
    q_max = wp.vec3(wp.max(q[0], 0.0), wp.max(q[1], 0.0), wp.max(q[2], 0.0))
    outside = wp.length(q_max)
    inside = wp.min(wp.max(q[0], wp.max(q[1], q[2])), 0.0)
    return outside + inside


@wp.kernel
def sphere_sphere_collision_kernel(
    sphere_centers: wp.array(dtype=wp.vec3),
    sphere_velocities: wp.array(dtype=wp.vec3),
    sphere_radii: wp.array(dtype=float),
    sphere_inv_masses: wp.array(dtype=float),
    contact_offset: float,
    max_push: float,
    max_impulse: float,
):
    """Resolve mutual collision between the two spheres (translation-only).

    - Frictionless, inelastic normal response (restitution = 0).
    - Push-out is split by inverse masses (PBD-style).
    - Impulse is clamped optionally by max_impulse.
    """

    # Only one pair exists when NUM_SPHERES == 2; use single thread.
    if wp.tid() != 0:
        return

    c0 = sphere_centers[0]
    c1 = sphere_centers[1]
    v0 = sphere_velocities[0]
    v1 = sphere_velocities[1]
    r0 = sphere_radii[0]
    r1 = sphere_radii[1]
    w0 = sphere_inv_masses[0]
    w1 = sphere_inv_masses[1]

    diff = c1 - c0
    dist = wp.length(diff)
    target = r0 + r1 + contact_offset
    penetration = target - dist

    if penetration > 0.0:
        n = safe_normal(diff)

        wsum = w0 + w1
        if wsum > 0.0:
            push = penetration
            if max_push > 0.0 and push > max_push:
                push = max_push

            # Split push-out by inverse masses
            if w0 > 0.0:
                c0 = c0 - n * (push * (w0 / wsum))
            if w1 > 0.0:
                c1 = c1 + n * (push * (w1 / wsum))

            # Inelastic normal response
            v_rel_n = wp.dot(v1 - v0, n)
            if v_rel_n < 0.0:
                J = -v_rel_n / wsum  # scalar impulse magnitude
                Jvec = n * J

                # Optional clamp on impulse magnitude
                if max_impulse > 0.0:
                    Jmag = wp.length(Jvec)
                    if Jmag > max_impulse and Jmag > 0.0:
                        Jvec = Jvec * (max_impulse / Jmag)

                if w0 > 0.0:
                    v0 = v0 - Jvec * w0
                if w1 > 0.0:
                    v1 = v1 + Jvec * w1

    sphere_centers[0] = c0
    sphere_centers[1] = c1
    sphere_velocities[0] = v0
    sphere_velocities[1] = v1


@wp.kernel
def sphere_water_coupling_kernel(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    sphere_centers: wp.array(dtype=wp.vec3),
    sphere_velocities: wp.array(dtype=wp.vec3),
    sphere_radii: wp.array(dtype=float),
    sphere_inv_masses: wp.array(dtype=float),
    impulse_x: wp.array(dtype=float),
    impulse_y: wp.array(dtype=float),
    impulse_z: wp.array(dtype=float),
    particle_mass: float,
    particle_radius: float,
    contact_offset: float,
    max_push: float,
    max_impulse: float,
):
    """Project particles out of spheres and transfer momentum (two-way coupling).

    - No friction, no restitution (inelastic along normal).
    - Sphere is translation-only, so accumulated impulse updates sphere velocity.
    """

    i = wp.tid()
    p = positions[i]
    v = velocities[i]

    # Loop over a fixed number of spheres.
    for s in range(NUM_SPHERES):
        c = sphere_centers[s]
        R = sphere_radii[s] + particle_radius + contact_offset

        r = p - c
        dist = wp.length(r)
        penetration = R - dist

        if penetration > 0.0:
            n = safe_normal(r)

            # Bounded push-out (common PBD stability technique).
            push = penetration
            if push > max_push:
                push = max_push

            p = p + push * n

            # Inelastic normal response in relative velocity.
            vs = sphere_velocities[s]
            v_rel_n = wp.dot(v - vs, n)
            if v_rel_n < 0.0:
                v_new = v - v_rel_n * n
                dv = v_new - v

                Jx = particle_mass * dv[0]
                Jy = particle_mass * dv[1]
                Jz = particle_mass * dv[2]

                # Optional clamp on impulse magnitude per contact.
                Jmag = wp.sqrt(Jx * Jx + Jy * Jy + Jz * Jz)
                if Jmag > max_impulse and Jmag > 0.0:
                    scale = max_impulse / Jmag
                    Jx = Jx * scale
                    Jy = Jy * scale
                    Jz = Jz * scale
                    dv = wp.vec3(Jx / particle_mass, Jy / particle_mass, Jz / particle_mass)
                    v_new = v + dv

                v = v_new

                # Apply equal-and-opposite impulse onto sphere.
                if sphere_inv_masses[s] > 0.0:
                    wp.atomic_add(impulse_x, s, -Jx)
                    wp.atomic_add(impulse_y, s, -Jy)
                    wp.atomic_add(impulse_z, s, -Jz)

    positions[i] = p
    velocities[i] = v


@wp.kernel
def integrate_spheres_kernel(
    sphere_centers: wp.array(dtype=wp.vec3),
    sphere_velocities: wp.array(dtype=wp.vec3),
    sphere_inv_masses: wp.array(dtype=float),
    impulse_x: wp.array(dtype=float),
    impulse_y: wp.array(dtype=float),
    impulse_z: wp.array(dtype=float),
    dt: float,
    gravity: wp.vec3,
):
    """Integrate sphere centers and velocities, applying accumulated impulses.

    Impulse accumulators are reset to zero each call.
    """

    s = wp.tid()

    inv_m = sphere_inv_masses[s]
    v = sphere_velocities[s]

    if inv_m > 0.0:
        J = wp.vec3(impulse_x[s], impulse_y[s], impulse_z[s])
        v = v + inv_m * J

        # Gravity for spheres (needed for sink/float behavior).
        v = v + gravity * dt

        sphere_centers[s] = sphere_centers[s] + v * dt
        sphere_velocities[s] = v

    # reset impulse accumulators
    impulse_x[s] = 0.0
    impulse_y[s] = 0.0
    impulse_z[s] = 0.0


@wp.kernel
def sphere_container_collision_kernel(
    sphere_centers: wp.array(dtype=wp.vec3),
    sphere_velocities: wp.array(dtype=wp.vec3),
    translation: wp.vec3,
    half_extents: wp.vec3,
    sphere_radii: wp.array(dtype=float),
    contact_offset: float,
    max_push: float,
):
    """Keep sphere centers inside the (translated) axis-aligned box container.

    This is a translation-only analog of `container_collision_kernel`, but applied
    to sphere centers with radius inflation.
    """

    s = wp.tid()
    c_world = sphere_centers[s]
    v = sphere_velocities[s]

    # Convert to container-local coordinates.
    c = c_world - translation

    # Effective inner half extents (shrink by sphere radius + offset).
    R = sphere_radii[s] + contact_offset
    he = wp.vec3(half_extents[0] - R, half_extents[1] - R, half_extents[2] - R)

    # If radius is too large, avoid NaNs.
    he = wp.vec3(wp.max(he[0], 0.0), wp.max(he[1], 0.0), wp.max(he[2], 0.0))

    # Signed distance to box (same convention: negative inside, positive outside).
    dist = sdf_box_local(c, he)

    if dist > 0.0:
        # Estimate normal via finite differences (cheap at 2 spheres).
        eps = 1.0e-3
        dx = sdf_box_local(c + wp.vec3(eps, 0.0, 0.0), he) - sdf_box_local(c - wp.vec3(eps, 0.0, 0.0), he)
        dy = sdf_box_local(c + wp.vec3(0.0, eps, 0.0), he) - sdf_box_local(c - wp.vec3(0.0, eps, 0.0), he)
        dz = sdf_box_local(c + wp.vec3(0.0, 0.0, eps), he) - sdf_box_local(c - wp.vec3(0.0, 0.0, eps), he)
        n = safe_normal(wp.vec3(dx, dy, dz) * (0.5 / eps))

        push = dist
        if push > max_push:
            push = max_push

        c_world = c_world - push * n

        # Remove velocity component pointing outward.
        vn = wp.dot(v, n)
        if vn > 0.0:
            v = v - vn * n

    sphere_centers[s] = c_world
    sphere_velocities[s] = v
