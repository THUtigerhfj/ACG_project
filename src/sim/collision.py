"""Container SDF utilities and Warp kernels for boundary handling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import warp as wp


@dataclass
class ContainerState:
    half_extents: Tuple[float, float, float]
    translation: Tuple[float, float, float]
    wall_thickness: float
    device: str = "cuda"

    def __post_init__(self) -> None:
        self.translation_vec = wp.vec3(*self.translation) # * pass the tuple as three separate arguments
        self.half_extents_vec = wp.vec3(*self.half_extents)

    def set_translation(self, translation: Tuple[float, float, float]) -> None:
        self.translation = translation
        self.translation_vec = wp.vec3(*translation)

    def translation_wp(self) -> wp.vec3:
        return self.translation_vec

    def half_extents_wp(self) -> wp.vec3:
        return self.half_extents_vec


@wp.func
def signed_distance_box(p: wp.vec3, half_extents: wp.vec3) -> float:
    q = wp.vec3(
        wp.abs(p[0]) - half_extents[0],
        wp.abs(p[1]) - half_extents[1],
        wp.abs(p[2]) - half_extents[2],
    )
    q_max = wp.vec3(wp.max(q[0], 0.0), wp.max(q[1], 0.0), wp.max(q[2], 0.0))
    outside = wp.length(q_max)
    inside = wp.min(wp.max(q[0], wp.max(q[1], q[2])), 0.0)
    return outside + inside


@wp.func
def estimate_gradient(p: wp.vec3, half_extents: wp.vec3, eps: float = 1e-3) -> wp.vec3:
    dx = signed_distance_box(p + wp.vec3(eps, 0.0, 0.0), half_extents) - signed_distance_box(p - wp.vec3(eps, 0.0, 0.0), half_extents)
    dy = signed_distance_box(p + wp.vec3(0.0, eps, 0.0), half_extents) - signed_distance_box(p - wp.vec3(0.0, eps, 0.0), half_extents)
    dz = signed_distance_box(p + wp.vec3(0.0, 0.0, eps), half_extents) - signed_distance_box(p - wp.vec3(0.0, 0.0, eps), half_extents)
    grad = wp.vec3(dx, dy, dz) * (0.5 / eps)
    length = wp.length(grad)
    if length > 0.0:
        grad = grad / length
    return grad


@wp.kernel
def container_collision_kernel(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    translation: wp.vec3,
    half_extents: wp.vec3,
):
    """Keep particles inside the container.
    
    half_extents defines the inner boundary of the container.
    SDF is negative inside, positive outside.
    Push particles back inside when SDF > 0.
    """
    i = wp.tid()
    p = positions[i] - translation
    dist = signed_distance_box(p, half_extents)
    
    # If particle is outside (dist > 0), push it back inside
    if dist > 0.0:
        n = estimate_gradient(p, half_extents)
        # Push particle inside along the inward normal (-n)
        positions[i] = positions[i] - dist * n
        # Remove velocity component pointing outward
        vn = wp.dot(velocities[i], n)
        if vn > 0.0:
            velocities[i] = velocities[i] - vn * n
