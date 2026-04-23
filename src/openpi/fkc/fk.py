"""Forward kinematics for Franka Panda in pure JAX.

Answers the "joint positions -> world EE position" question at inference time,
so the FKC cost function can work in task space while remaining fully
differentiable via ``jax.grad``.

Implementation: modified Denavit-Hartenberg (DH) convention using 4x4
homogeneous matrices. Parameters taken from the standard Franka Panda URDF
(frankaemika/franka_ros/franka_description) and match what RoboLab ships in
``franka_robotiq_2f_85_flattened.usd``.

No external kinematics library needed — this keeps the server's dep list clean
and the whole FK traceable/JIT-able under JAX.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

# Modified DH parameters for Franka Panda's 7-DoF arm.
# Each row: (a_{i-1}, alpha_{i-1}, d_i, theta_offset_i)
# i.e. the parameters that describe the transform from frame i-1 to frame i.
# Values sourced from the Franka URDF. The standard panda_link8 flange offset
# (d=0.107) is folded into the chain below so the output frame is panda_link8.
_FRANKA_MDH: tuple[tuple[float, float, float, float], ...] = (
    (0.0, 0.0, 0.333, 0.0),
    (0.0, -jnp.pi / 2, 0.0, 0.0),
    (0.0, jnp.pi / 2, 0.316, 0.0),
    (0.0825, jnp.pi / 2, 0.0, 0.0),
    (-0.0825, -jnp.pi / 2, 0.384, 0.0),
    (0.0, jnp.pi / 2, 0.0, 0.0),
    (0.088, jnp.pi / 2, 0.0, 0.0),
)
# Fixed flange->panda_link8 transform (pure translation along +z).
_FLANGE_OFFSET_Z: float = 0.107


def _mdh_transform(a: float, alpha: float, d: float, theta: jnp.ndarray) -> jnp.ndarray:
    """Single modified-DH link transform. Returns a (4,4) matrix."""
    ct = jnp.cos(theta)
    st = jnp.sin(theta)
    ca = jnp.cos(alpha)
    sa = jnp.sin(alpha)
    return jnp.array(
        [
            [ct, -st, 0.0, a],
            [st * ca, ct * ca, -sa, -d * sa],
            [st * sa, ct * sa, ca, d * ca],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def _quat_xyzw_to_matrix(q: tuple[float, float, float, float]) -> jnp.ndarray:
    x, y, z, w = q
    return jnp.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w), 0.0],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w), 0.0],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y), 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def make_static_transform(xyz: tuple[float, float, float], quat_xyzw: tuple[float, float, float, float]) -> jnp.ndarray:
    T = _quat_xyzw_to_matrix(quat_xyzw)
    return T.at[:3, 3].set(jnp.asarray(xyz, dtype=jnp.float32))


def franka_fk(
    joint_pos_7: jnp.ndarray,
    base_world_T: jnp.ndarray,
    ee_offset_T: jnp.ndarray,
) -> jnp.ndarray:
    """Forward kinematics. Returns the full (4,4) world-frame EE transform.

    Args:
        joint_pos_7: (7,) Franka arm joint angles in radians, absolute.
        base_world_T: (4,4) robot base pose in world frame.
        ee_offset_T: (4,4) fixed offset from panda_link8 to the target EE body.

    The full chain is base_world @ link_1 @ ... @ link_7 @ flange_offset @ ee_offset.
    """
    T = base_world_T
    for i, (a, alpha, d, theta_off) in enumerate(_FRANKA_MDH):
        T = T @ _mdh_transform(a, alpha, d, joint_pos_7[i] + theta_off)
    # Flange offset along +z (panda_link7 -> panda_link8).
    flange_offset = jnp.eye(4).at[2, 3].set(_FLANGE_OFFSET_Z)
    T = T @ flange_offset @ ee_offset_T
    return T


def franka_fk_position(joint_pos_7: jnp.ndarray, base_world_T: jnp.ndarray, ee_offset_T: jnp.ndarray) -> jnp.ndarray:
    """Convenience: return just the (3,) world-frame EE position."""
    return franka_fk(joint_pos_7, base_world_T, ee_offset_T)[:3, 3]


def build_fk_transforms(fk_cfg: Any) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Materialise the two static SE3 matrices from an ``FKConfig``.

    Typed as ``Any`` to avoid an import cycle with ``openpi.fkc.config``.
    """
    return (
        make_static_transform(fk_cfg.base_xyz, fk_cfg.base_quat_xyzw),
        make_static_transform(fk_cfg.ee_offset_xyz, fk_cfg.ee_offset_quat_xyzw),
    )
