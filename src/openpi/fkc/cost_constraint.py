"""Placeholder cost L(x), constraints h_eq / h_ineq, and the composite J(x).

Everything here is deliberately lightweight — the user will swap in real
task-specific cost/constraint functions later. The scaffolding is what matters
for now:

* ``J_value`` applies the *value* weight set ``w_*_value`` from ``FKCConfig``.
* ``J_grad`` applies the *gradient* weight set ``w_*_grad`` and returns
  ∇_actions of the weighted objective via ``jax.grad``.

Those two distinct weight sets are the user-requested decoupling: ``J_value``
enters the Feynman-Kac log-weight update, while ``J_grad`` enters the augmented
drift. In gr00t's original implementation both came from the same weighted J.

Current placeholders:
    - L(path):  squared distance of the final EE position to ``target_xyz``.
    - h_eq:     none for now (returns a scalar 0 tensor so sums work).
    - h_ineq:   axis-aligned bounding-box penalty, softplus-smoothed.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.nn as jnn
import jax.numpy as jnp

from openpi.fkc.path import build_world_path


class FKRuntime(NamedTuple):
    """JAX-friendly bundle of the per-rollout FK + normalization context.

    Rebuilt once per ``Policy.infer`` call and passed through the sampling loop
    as a leaf-traced structure (it's a PyTree of jnp arrays so jit is happy).
    """

    action_mean_7: jnp.ndarray    # (7,)
    action_std_7: jnp.ndarray     # (7,)
    current_joint_pos: jnp.ndarray  # (B, 7) absolute radians
    base_world_T: jnp.ndarray     # (4,4)
    ee_offset_T: jnp.ndarray      # (4,4)

    # Cost / constraint parameters, pre-materialised as jnp arrays.
    target_xyz: jnp.ndarray       # (3,)
    box_min_xyz: jnp.ndarray      # (3,)
    box_max_xyz: jnp.ndarray      # (3,)
    softplus_beta: jnp.ndarray    # scalar


def _repeat_runtime_for_particles(rt: FKRuntime, num_particles: int) -> FKRuntime:
    """Tile ``current_joint_pos`` across the particle dimension."""
    if num_particles == 1:
        return rt
    cjp = rt.current_joint_pos
    tiled = jnp.repeat(cjp, num_particles, axis=0)
    return rt._replace(current_joint_pos=tiled)


def _L_final_distance(world_path: jnp.ndarray, rt: FKRuntime) -> jnp.ndarray:
    """Placeholder cost: squared distance of last EE waypoint to a target."""
    last = world_path[..., -1, :]  # (B, 3)
    diff = last - rt.target_xyz
    return jnp.sum(diff * diff, axis=-1)  # (B,)


def _h_eq(world_path: jnp.ndarray, rt: FKRuntime) -> jnp.ndarray:
    """Placeholder equality residual — none active. Returns (B, 1) of zeros so
    the squared sum is well-defined and JIT-traceable. Replace later."""
    del rt
    b = world_path.shape[0]
    return jnp.zeros((b, 1), dtype=world_path.dtype)


def _h_ineq(world_path: jnp.ndarray, rt: FKRuntime) -> jnp.ndarray:
    """Placeholder inequality residual: keep every EE waypoint inside a box.

    Positive values mean a violation; softplus makes the hinge smooth so the
    gradient is non-zero on the boundary.
    """
    below = rt.box_min_xyz - world_path  # (B, H, 3), >0 when violated
    above = world_path - rt.box_max_xyz
    violation = jnp.concatenate([below, above], axis=-1)  # (B, H, 6)
    beta = rt.softplus_beta
    # softplus(beta * v) / beta — smooth max(0, v).
    # return jnn.softplus(beta * violation) / beta
    def _print_violation(v):
        jax.debug.print(
            "violation min={} max={} shape={}",
            jnp.min(v),
            jnp.max(v),
            v.shape,
        )
        return ()

    jax.lax.cond(
        jnp.max(violation) >= 0,
        _print_violation,
        lambda _: (),
        violation,
    )

    return jnp.maximum(violation, 0.0)


def _weighted_objective(
    actions: jnp.ndarray,
    rt: FKRuntime,
    *,
    w_cost: float,
    w_eq: float,
    w_ineq: float,
) -> jnp.ndarray:
    """Weighted scalar objective. Shape: (B,)."""
    path = build_world_path(
        actions,
        rt.current_joint_pos,
        rt.action_mean_7,
        rt.action_std_7,
        rt.base_world_T,
        rt.ee_offset_T,
    )
    cost_term = w_cost * _L_final_distance(path, rt)
    eq_term = w_eq * jnp.sum(jnp.square(_h_eq(path, rt)), axis=-1)
    ineq_term = w_ineq * jnp.sum(jnp.square(_h_ineq(path, rt)), axis=(-1, -2))
    return cost_term + eq_term + ineq_term


def J_value(actions: jnp.ndarray, rt: FKRuntime, fkc_cfg: Any) -> jnp.ndarray:
    """J(x) that goes into the FKC log-weight update."""
    return _weighted_objective(
        actions,
        rt,
        w_cost=fkc_cfg.w_cost_value,
        w_eq=fkc_cfg.w_eq_value,
        w_ineq=fkc_cfg.w_ineq_value,
    )


def _scalar_J_grad(actions: jnp.ndarray, rt: FKRuntime, fkc_cfg: Any) -> jnp.ndarray:
    per_sample = _weighted_objective(
        actions,
        rt,
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
