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


def _chain_link_transforms(joint_pos_7: jnp.ndarray, base_world_T: jnp.ndarray) -> jnp.ndarray:
    """Returns ``(8, 4, 4)``: ``[base, link1, ..., link7]`` world-frame transforms.

    The i-th entry (1 ≤ i ≤ 7) is the world-frame pose of the panda_link``i``
    frame after applying joints 1..i. Entry 0 is the base (panda_link0).
    """
    transforms = [base_world_T]
    T = base_world_T
    for i, (a, alpha, d, theta_off) in enumerate(_FRANKA_MDH):
        T = T @ _mdh_transform(a, alpha, d, joint_pos_7[i] + theta_off)
        transforms.append(T)
    return jnp.stack(transforms, axis=0)


def _flange_and_ee_transforms(link7_T: jnp.ndarray, ee_offset_T: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    flange_offset = jnp.eye(4, dtype=link7_T.dtype).at[2, 3].set(_FLANGE_OFFSET_Z)
    T_flange = link7_T @ flange_offset
    T_ee = T_flange @ ee_offset_T
    return T_flange, T_ee


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
    chain = _chain_link_transforms(joint_pos_7, base_world_T)
    _, T_ee = _flange_and_ee_transforms(chain[-1], ee_offset_T)
    return T_ee


def franka_fk_position(joint_pos_7: jnp.ndarray, base_world_T: jnp.ndarray, ee_offset_T: jnp.ndarray) -> jnp.ndarray:
    """Convenience: return just the (3,) world-frame EE position."""
    return franka_fk(joint_pos_7, base_world_T, ee_offset_T)[:3, 3]


# Local-frame offsets for the gripper finger / palm sample points, expressed
# in the EE frame (after ``ee_offset_T`` has been applied). The Robotiq 2F-85
# fully-open finger separation is ~85 mm, tips reach ~6 cm forward of the
# flange. These three points cover the gripper body volume coarsely.
_GRIPPER_LOCAL_POINTS: tuple[tuple[float, float, float], ...] = (
    (0.0, -0.040, 0.060),  # left finger tip
    (0.0,  0.040, 0.060),  # right finger tip
    (0.0,  0.000, 0.020),  # palm
)


def _polyline_arc_samples(polyline: jnp.ndarray, n: int) -> jnp.ndarray:
    """Sample ``n`` points along a polyline at uniform arc length.

    polyline: ``(K, 3)`` of K joint origins through the kinematic chain.
    Returns ``(n, 3)``. Pure JAX, ``jit``/``grad`` friendly.
    """
    seg_vecs = polyline[1:] - polyline[:-1]
    seg_lens = jnp.sqrt(jnp.sum(seg_vecs * seg_vecs, axis=-1) + 1e-12)  # (K-1,)
    cum_lens = jnp.concatenate(
        [jnp.zeros((1,), dtype=seg_lens.dtype), jnp.cumsum(seg_lens)],
        axis=0,
    )  # (K,)
    total_len = cum_lens[-1]
    # Uniform fractions in [0, 1] mapped to arc length.
    fractions = jnp.linspace(0.0, 1.0, n, dtype=polyline.dtype)
    sample_t = fractions * total_len  # (n,)
    # Locate the segment containing each sample. searchsorted returns the
    # rightmost insertion index; we subtract 1 to get the segment index, and
    # clip to [0, K-2] so endpoints behave (the last point lands at the end
    # of the final segment).
    raw_idx = jnp.searchsorted(cum_lens, sample_t)
    seg_idx = jnp.clip(raw_idx - 1, 0, polyline.shape[0] - 2)
    seg_starts = polyline[seg_idx]
    seg_ends = polyline[seg_idx + 1]
    seg_local = (sample_t - cum_lens[seg_idx]) / (seg_lens[seg_idx] + 1e-12)
    return seg_starts + seg_local[:, None] * (seg_ends - seg_starts)


def franka_collision_points(
    joint_pos_7: jnp.ndarray,
    base_world_T: jnp.ndarray,
    ee_offset_T: jnp.ndarray,
    *,
    mode: str,
    arm_sample_points: int = 8,
    full_body_points: int = 30,
) -> jnp.ndarray:
    """Sample world-frame points along the robot for SDF collision queries.

    The number of points returned depends on ``mode`` and the count knobs:

    * ``"ee_only"`` -> ``(1, 3)``: just the EE position (matches
      ``franka_fk_position``).
    * ``"ee_plus_arm"`` -> ``(arm_sample_points + 4, 3)``: EE + 3 gripper points
      (two finger tips + palm) + ``arm_sample_points`` points spaced uniformly
      along the kinematic-chain polyline (link1 -> ... -> link7 -> flange ->
      ee). Default ``arm_sample_points=8`` gives 12 points.
    * ``"full"`` -> ``(full_body_points + 4, 3)``: same structure but with
      ``full_body_points`` along the arm. Default 30 gives 34 points.

    All counts are static at JIT trace time because ``mode`` and the integer
    knobs are read from ``FKCConfig`` (a static argument). ``jit`` and ``grad``
    propagate cleanly through the polyline sampling.
    """
    chain = _chain_link_transforms(joint_pos_7, base_world_T)  # (8, 4, 4)
    T_flange, T_ee = _flange_and_ee_transforms(chain[-1], ee_offset_T)
    ee_origin = T_ee[:3, 3]

    if mode == "ee_only":
        return ee_origin[None, :]

    # Gripper finger / palm points: transform local offsets through T_ee.
    local = jnp.asarray(_GRIPPER_LOCAL_POINTS, dtype=base_world_T.dtype)  # (3, 3)
    local_h = jnp.concatenate([local, jnp.ones((local.shape[0], 1), dtype=local.dtype)], axis=-1)  # (3, 4)
    gripper_world = (T_ee @ local_h.T)[:3, :].T  # (3, 3)

    # Polyline through link1..link7, flange, ee — the arm physical centerline.
    polyline = jnp.concatenate(
        [chain[1:, :3, 3], T_flange[:3, 3][None, :], ee_origin[None, :]], axis=0
    )  # (9, 3)

    if mode == "ee_plus_arm":
        n_arm = int(arm_sample_points)
    elif mode == "full":
        n_arm = int(full_body_points)
    else:
        raise ValueError(
            f"Unknown collision mode: {mode!r}. Expected 'ee_only', 'ee_plus_arm', or 'full'."
        )

    arm_samples = _polyline_arc_samples(polyline, n_arm)  # (n_arm, 3)
    return jnp.concatenate(
        [ee_origin[None, :], gripper_world, arm_samples],
        axis=0,
    )  # (n_arm + 4, 3)


def collision_points_count(mode: str, arm_sample_points: int = 8, full_body_points: int = 30) -> int:
    """Static count of collision points for a given mode.

    Useful for shape inference outside JIT.
    """
    if mode == "ee_only":
        return 1
    if mode == "ee_plus_arm":
        return int(arm_sample_points) + 4
    if mode == "full":
        return int(full_body_points) + 4
    raise ValueError(
        f"Unknown collision mode: {mode!r}. Expected 'ee_only', 'ee_plus_arm', or 'full'."
    )


def build_fk_transforms(fk_cfg: Any) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Materialise the two static SE3 matrices from an ``FKConfig``.

    Typed as ``Any`` to avoid an import cycle with ``openpi.fkc.config``.
    """
    return (
        make_static_transform(fk_cfg.base_xyz, fk_cfg.base_quat_xyzw),
        make_static_transform(fk_cfg.ee_offset_xyz, fk_cfg.ee_offset_quat_xyzw),
    )
