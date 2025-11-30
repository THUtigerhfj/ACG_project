"""Warp kernels for the SPH / PBF simulation stages."""

from __future__ import annotations

import math

import warp as wp


def _poly6_coeff(h: float) -> float:
    return 315.0 / (64.0 * math.pi * pow(h, 9))


def _spiky_grad_coeff(h: float) -> float:
    return -45.0 / (math.pi * pow(h, 6))


def create_kernel_params(smoothing_length: float) -> tuple[float, float]:
    """Precompute kernel constants reused across launches."""

    return _poly6_coeff(smoothing_length), _spiky_grad_coeff(smoothing_length)


@wp.func
def kernel_poly6(r: float, h: float, coeff: float) -> float:
    if r >= h:
        return 0.0
    x = h * h - r * r
    return coeff * x * x * x


@wp.func
def kernel_spiky_grad(pos_i: wp.vec3, pos_j: wp.vec3, h: float, coeff: float) -> wp.vec3:
    r = pos_i - pos_j
    dist = wp.length(r)
    if dist == 0.0 or dist >= h:
        return wp.vec3()
    grad = coeff * pow(h - dist, 2.0) * (r / dist)
    return grad


@wp.kernel
def predict_positions_kernel(
    positions: wp.array(dtype=wp.vec3),
    prev_positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    dt: float,
    gravity: wp.vec3,
):
    i = wp.tid()
    prev_positions[i] = positions[i]
    vi = velocities[i] + gravity * dt
    positions[i] = positions[i] + vi * dt
    velocities[i] = vi


@wp.kernel
def density_kernel(
    positions: wp.array(dtype=wp.vec3),
    densities: wp.array(dtype=float),
    mass: float,
    h: float,
    poly6_coeff: float,
    grid_id: wp.uint64,
):
    i = wp.tid()
    pos_i = positions[i]
    rho = float(0.0)
    query = wp.hash_grid_query(grid_id, pos_i, h)
    j = int(0)
    while wp.hash_grid_query_next(query, j):
        r = wp.length(pos_i - positions[j])
        rho += mass * kernel_poly6(r, h, poly6_coeff)
    densities[i] = rho


@wp.kernel
def lambda_kernel(
    positions: wp.array(dtype=wp.vec3),
    densities: wp.array(dtype=float),
    lambdas: wp.array(dtype=float),
    mass: float,
    rest_density: float,
    h: float,
    spiky_coeff: float,
    eps: float,
    grid_id: wp.uint64,
):
    i = wp.tid()
    pos_i = positions[i]
    rho_i = densities[i]
    # Clamp constraint to only handle compression (not expansion)
    # This prevents boundary particles with low density from being pushed outward
    Ci = wp.max(rho_i / rest_density - 1.0, 0.0)
    grad_sum = float(0.0)
    grad_i = wp.vec3(0.0, 0.0, 0.0)

    query = wp.hash_grid_query(grid_id, pos_i, h)
    j = int(0)
    while wp.hash_grid_query_next(query, j):
        if j == i:
            continue
        grad = (mass / rest_density) * kernel_spiky_grad(pos_i, positions[j], h, spiky_coeff)
        grad_sum += wp.dot(grad, grad)
        grad_i += grad

    grad_sum += wp.dot(grad_i, grad_i)
    lambdas[i] = -Ci / (grad_sum + eps)


@wp.kernel
def delta_kernel(
    positions: wp.array(dtype=wp.vec3),
    lambdas: wp.array(dtype=float),
    delta_pos: wp.array(dtype=wp.vec3),
    mass: float,
    rest_density: float,
    h: float,
    spiky_coeff: float,
    grid_id: wp.uint64,
):
    i = wp.tid()
    pos_i = positions[i]
    lambda_i = lambdas[i]
    delta = wp.vec3(0.0, 0.0, 0.0)
    query = wp.hash_grid_query(grid_id, pos_i, h)
    j = int(0)
    while wp.hash_grid_query_next(query, j):
        if j == i:
            continue
        lambda_sum = lambda_i + lambdas[j]
        grad = kernel_spiky_grad(pos_i, positions[j], h, spiky_coeff)
        delta += (mass / rest_density) * lambda_sum * grad
    delta_pos[i] = delta


@wp.kernel
def apply_delta_kernel(
    positions: wp.array(dtype=wp.vec3),
    delta_pos: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    positions[i] = positions[i] + delta_pos[i]
    delta_pos[i] = wp.vec3()


@wp.kernel
def update_velocity_kernel(
    positions: wp.array(dtype=wp.vec3),
    prev_positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    inv_dt: float,
):
    i = wp.tid()
    velocities[i] = (positions[i] - prev_positions[i]) * inv_dt


@wp.kernel
def copy_vec3_array_kernel(
    src: wp.array(dtype=wp.vec3),
    dst: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    dst[i] = src[i]


@wp.kernel
def xsph_kernel(
    positions: wp.array(dtype=wp.vec3),
    velocities_in: wp.array(dtype=wp.vec3),
    velocities_out: wp.array(dtype=wp.vec3),
    h: float,
    xsph_c: float,
    poly6_coeff: float,
    grid_id: wp.uint64,
):
    """XSPH viscosity kernel using normalized weighted average.
    
    Uses weighted sum of neighbor velocities instead of density-based weighting.
    This avoids numerical issues with low-density boundary particles.
    """
    i = wp.tid()
    pos_i = positions[i]
    vi = velocities_in[i]
    
    wsum = float(0.0)
    v_acc = wp.vec3(0.0, 0.0, 0.0)
    
    query = wp.hash_grid_query(grid_id, pos_i, h)
    j = int(0)
    while wp.hash_grid_query_next(query, j):
        if j == i:
            continue
        r = wp.length(pos_i - positions[j])
        w = kernel_poly6(r, h, poly6_coeff)
        wsum += w
        v_acc += velocities_in[j] * w
    
    # Normalized weighted average of neighbor velocities
    if wsum > 1e-8:
        velocities_out[i] = (1.0 - xsph_c) * vi + xsph_c * (v_acc / wsum)
    else:
        velocities_out[i] = vi