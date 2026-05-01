"""Cost ``L(x)``, equality / inequality residuals, and the composite ``J(x)``.

The inequality term is an **SDF-based collision avoidance penalty** built from
a voxelised Euclidean Signed Distance Field (ESDF) of the scene. The ESDF is
constructed once per ``Policy.infer`` call by RoboLab's nvblox sidecar, shipped
through the websocket as ``fkc/sdf_*`` keys, and threaded into the ``FKRuntime``
NamedTuple — the constraint side never imports torch / nvblox.

Wiring per call:

* ``J_value`` applies the *value* weight set ``w_*_value`` from ``FKCConfig``.
  This enters the Feynman-Kac log-weight update.
* ``J_grad`` applies the *gradient* weight set ``w_*_grad`` and returns
  ``∇_actions`` of the weighted objective via ``jax.grad``. This enters the
  augmented drift in ``sample_actions_guided``.

Currently:
    - L(path):   returns 0 (placeholder; the user's intent is to replace later).
    - h_eq:      returns zeros (no equality constraints active).
    - h_ineq:    SDF collision penalty across all FK collision points and all
                 horizon steps.

When the SDF grid is absent (e.g., baseline/no-RoboLab runs), a degenerate
1×1×1 grid filled with a large positive value is shipped via the no-op fallback
in ``policy.py`` — this means every collision query returns
``sdf >> safety_margin`` and the penalty is 0 everywhere, so the FKC pipeline
stays wired up but contributes nothing.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.nn as jnn
import jax.numpy as jnp

from openpi.fkc.path import build_world_collision_path


class FKRuntime(NamedTuple):
    """JAX-friendly bundle of the per-rollout FK + SDF context.

    Rebuilt once per ``Policy.infer`` call and passed through the sampling loop
    as a leaf-traced structure (it's a PyTree of jnp arrays so jit is happy).

    The SDF lives on the runtime (rather than the static FKCConfig) because
    the grid contents change with every replan but the *shape* stays fixed —
    so JIT recompiles only once.
    """

    action_mean_7: jnp.ndarray    # (7,)
    action_std_7: jnp.ndarray     # (7,)
    current_joint_pos: jnp.ndarray  # (B, 7) absolute radians
    current_joint_vel: jnp.ndarray  # (B, 7) absolute rad/s — initial v for actuator rollout
    base_world_T: jnp.ndarray     # (4,4)
    ee_offset_T: jnp.ndarray      # (4,4)

    # Reserved for a future terminal cost. Kept on the runtime for symmetry
    # with the constraint params even though ``_L_zero`` doesn't read it.
    target_xyz: jnp.ndarray       # (3,)

    # SDF — the heart of the constraint.
    sdf_grid: jnp.ndarray         # (Nx, Ny, Nz) float32, positive outside obstacles
    sdf_origin: jnp.ndarray       # (3,) world-frame position of voxel (0,0,0)
    sdf_voxel_size: jnp.ndarray   # scalar float32, edge length of one voxel (m)

    # Constraint hinge parameters.
    safety_margin: jnp.ndarray    # scalar float32, hinge fires when sdf < margin
    softplus_beta: jnp.ndarray    # scalar float32, hinge sharpness


def _repeat_runtime_for_particles(rt: FKRuntime, num_particles: int) -> FKRuntime:
    """Tile per-batch fields across particles. SDF / static fields are shared."""
    if num_particles == 1:
        return rt
    return rt._replace(
        current_joint_pos=jnp.repeat(rt.current_joint_pos, num_particles, axis=0),
        current_joint_vel=jnp.repeat(rt.current_joint_vel, num_particles, axis=0),
    )


def _L_zero(world_path: jnp.ndarray, rt: FKRuntime) -> jnp.ndarray:
    """Placeholder cost: zero, with the right shape ``(B,)``.

    Kept as a function (rather than literal 0) so the existing weighted-sum
    scaffolding in ``_weighted_objective`` continues to compose cleanly when a
    real terminal cost is plugged in later.
    """
    del rt
    # world_path: (B, H, P, 3) — collision-path layout. Take batch axis.
    return jnp.zeros((world_path.shape[0],), dtype=world_path.dtype)


def _h_eq(world_path: jnp.ndarray, rt: FKRuntime) -> jnp.ndarray:
    """No equality constraints active. Returns ``(B, 1)`` of zeros so the
    squared sum is well-defined and traceable."""
    del rt
    return jnp.zeros((world_path.shape[0], 1), dtype=world_path.dtype)


def _sdf_trilinear(
    grid: jnp.ndarray,
    origin: jnp.ndarray,
    voxel_size: jnp.ndarray,
    points: jnp.ndarray,
) -> jnp.ndarray:
    """Trilinear interpolation of a regular 3D scalar field at world points.

    Args:
        grid: ``(Nx, Ny, Nz)`` SDF values in world frame. ``grid[i,j,k]`` is
            the value at world position ``origin + voxel_size * (i,j,k)``.
        origin: ``(3,)`` world-frame position of voxel ``(0, 0, 0)``.
        voxel_size: scalar — the edge length of a voxel in metres.
        points: ``(..., 3)`` world-frame query points.

    Returns:
        ``(...)`` interpolated SDF values. Out-of-grid points are clamped to
        the nearest in-grid voxel — for FKC this is fine because the grid is
        sized to comfortably contain the workspace and far-away points reading
        the boundary value (which is large/free) yields zero penalty.

    Pure JAX, fully ``jit`` and ``grad`` friendly: the trilinear weighting is
    smooth, so ``jax.grad`` propagates gradients to the query ``points``.
    """
    nx, ny, nz = grid.shape
    # Convert world coords to fractional voxel coords.
    grid_xyz = (points - origin) / voxel_size  # (..., 3)
    # Floor to integer voxel indices, clamping so the +1 neighbour is in-bounds.
    i0 = jnp.clip(jnp.floor(grid_xyz[..., 0]).astype(jnp.int32), 0, nx - 2)
    j0 = jnp.clip(jnp.floor(grid_xyz[..., 1]).astype(jnp.int32), 0, ny - 2)
    k0 = jnp.clip(jnp.floor(grid_xyz[..., 2]).astype(jnp.int32), 0, nz - 2)
    i1, j1, k1 = i0 + 1, j0 + 1, k0 + 1
    # Fractional weights, also clamped to [0, 1] so out-of-grid queries fall
    # back to the boundary voxel value (which represents "free space" given
    # how the grid is sized).
    fx = jnp.clip(grid_xyz[..., 0] - i0.astype(grid_xyz.dtype), 0.0, 1.0)
    fy = jnp.clip(grid_xyz[..., 1] - j0.astype(grid_xyz.dtype), 0.0, 1.0)
    fz = jnp.clip(grid_xyz[..., 2] - k0.astype(grid_xyz.dtype), 0.0, 1.0)
    # 8 corner samples.
    c000 = grid[i0, j0, k0]
    c001 = grid[i0, j0, k1]
    c010 = grid[i0, j1, k0]
    c011 = grid[i0, j1, k1]
    c100 = grid[i1, j0, k0]
    c101 = grid[i1, j0, k1]
    c110 = grid[i1, j1, k0]
    c111 = grid[i1, j1, k1]
    # Trilinear interpolation.
    c00 = c000 * (1 - fx) + c100 * fx
    c01 = c001 * (1 - fx) + c101 * fx
    c10 = c010 * (1 - fx) + c110 * fx
    c11 = c011 * (1 - fx) + c111 * fx
    c0 = c00 * (1 - fy) + c10 * fy
    c1 = c01 * (1 - fy) + c11 * fy
    return c0 * (1 - fz) + c1 * fz


def _sdf_collision_penalty(world_collision_path: jnp.ndarray, rt: FKRuntime) -> jnp.ndarray:
    """Softplus-smoothed hinge over ``safety_margin - sdf(p)`` for every point.

    Args:
        world_collision_path: ``(B, H, P, 3)`` world-frame collision-sample
            points along the predicted arm trajectory.
        rt: runtime carrying the SDF grid + origin + voxel_size + margin.

    Returns:
        ``(B,)`` sum of point-wise penalties across all H horizon steps and
        all P body points.
    """
    sdf_values = _sdf_trilinear(
        rt.sdf_grid, rt.sdf_origin, rt.sdf_voxel_size, world_collision_path
    )  # (B, H, P)
    excess = rt.safety_margin - sdf_values
    # softplus(beta * x) / beta — smooth max(0, x) hinge.
    beta = rt.softplus_beta
    penalty = jnn.softplus(beta * excess) / beta
    return jnp.sum(penalty, axis=(-2, -1))  # (B,)


def _weighted_objective(
    actions: jnp.ndarray,
    rt: FKRuntime,
    fkc_cfg: Any,
    *,
    w_cost: float,
    w_eq: float,
    w_ineq: float,
) -> jnp.ndarray:
    """Weighted scalar objective. Shape: (B,)."""
    path = build_world_collision_path(
        actions,
        rt.current_joint_pos,
        rt.current_joint_vel,
        rt.action_mean_7,
        rt.action_std_7,
        rt.base_world_T,
        rt.ee_offset_T,
        fkc_cfg.dynamics,
        collision_mode=fkc_cfg.collision.mode,
        arm_sample_points=fkc_cfg.collision.arm_sample_points,
        full_body_points=fkc_cfg.collision.full_body_points,
    )
    cost_term = w_cost * _L_zero(path, rt)
    eq_term = w_eq * jnp.sum(jnp.square(_h_eq(path, rt)), axis=-1)
    ineq_term = w_ineq * _sdf_collision_penalty(path, rt)
    return cost_term + eq_term + ineq_term


def J_value(actions: jnp.ndarray, rt: FKRuntime, fkc_cfg: Any) -> jnp.ndarray:
    """J(x) that goes into the FKC log-weight update."""
    return _weighted_objective(
        actions,
        rt,
        fkc_cfg,
        w_cost=fkc_cfg.w_cost_value,
        w_eq=fkc_cfg.w_eq_value,
        w_ineq=fkc_cfg.w_ineq_value,
    )


def _scalar_J_grad(actions: jnp.ndarray, rt: FKRuntime, fkc_cfg: Any) -> jnp.ndarray:
    per_sample = _weighted_objective(
        actions,
        rt,
        fkc_cfg,
        w_cost=fkc_cfg.w_cost_grad,
        w_eq=fkc_cfg.w_eq_grad,
        w_ineq=fkc_cfg.w_ineq_grad,
    )
    return jnp.sum(per_sample)


def J_value_and_grad(
    actions: jnp.ndarray,
    rt: FKRuntime,
    fkc_cfg: Any,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return (J_value(x), ∇_x J_grad(x)).

    Note that J_value and the cost whose gradient we differentiate use
    *different* weight sets. That is the user-requested decoupling — when the
    weight sets coincide the behavior matches gr00t's.
    """
    value = J_value(actions, rt, fkc_cfg)
    grad = jax.grad(_scalar_J_grad)(actions, rt, fkc_cfg)
    return value, grad
