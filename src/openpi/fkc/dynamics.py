"""Mini physics simulator for the Franka Panda + Robotiq 2F-85 in Isaac Lab.

Replaces the "perfect tracking" assumption in ``path.py`` with the same
implicit-PD law that PhysX runs inside Isaac Lab's ``ImplicitActuator``.
Each control step (one chunk action) is rolled forward through ``decimation``
physics substeps using the configuration-dependent mass matrix M(q).

Per-substep update (implicit Euler with PD, no gravity, no Coriolis):
    (M(q_k) + dt·D·I + dt²·K·I) v_{k+1} = M(q_k) v_k + dt·K·(q_cmd - q_k)
    q_{k+1} = q_k + dt·v_{k+1}

Modeling assumptions, all justified for ``franka_robotiq_2f_85_flattened.usd``
under ``DroidCfg``:

* ``disable_gravity=True`` on the robot rigid bodies → no g(q) term.
* Coriolis/centrifugal are negligible for q̇ < ~2 rad/s (chunk speeds).
* Effort limits are NOT enforced here. PhysX clips at ``effort_limit`` for the
  wrist joints (12 N·m); ignoring that makes our predictions slightly
  optimistic in fast wrist motions.
* Joint position limits are enforced as a hard clip after each substep, with
  velocity zeroed in the limit-violating direction (wall hit).

Body model: 7 Franka arm links with canonical inertias from
``franka_description/panda_arm.xacro`` plus one rigidly-attached "hand_lump"
body for the panda_hand + Robotiq 2F-85 assembly at panda_link8.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from openpi.fkc.fk import _FLANGE_OFFSET_Z, _FRANKA_MDH, _mdh_transform


# ---------------------------------------------------------------------------
# Inertial parameters (Franka URDF)
# Each tuple: (mass_kg, com_xyz_local, (ixx, iyy, izz, ixy, ixz, iyz)).
# ---------------------------------------------------------------------------
_PANDA_LINK_INERTIAS: tuple[tuple[float, tuple[float, float, float], tuple[float, ...]], ...] = (
    (4.970684, (0.003875, 0.002081, -0.04762),  (0.70337,  0.70661,  0.009117, -0.000139, 0.006772, 0.019169)),
    (0.646926, (-0.003141, -0.02872, 0.003495), (0.007962, 0.02811,  0.025995, -0.003925, 0.010254, 0.000704)),
    (3.228604, (0.027518, 0.039252, -0.066502), (0.037242, 0.036155, 0.01083,  -0.004761, -0.011396, -0.012805)),
    (3.587895, (-0.05317, 0.104419, 0.027454),  (0.025853, 0.019552, 0.028323,  0.007796, -0.001332, 0.008641)),
    (1.225946, (-0.011953, 0.041065, -0.038437),(0.035549, 0.029474, 0.008627, -0.002117, -0.004037, 0.000229)),
    (1.666555, (0.060149, -0.014117, -0.010517),(0.001964, 0.004354, 0.005433,  0.000109, -0.001158, 0.000341)),
    (0.735522, (0.010517, -0.004252, 0.061597), (0.012516, 0.010027, 0.004815, -0.000428, -0.001196, -0.000741)),
)

# Lumped panda_hand + Robotiq 2F-85, rigidly attached to panda_link8.
# Frame: panda_link8 (= panda_link7 + flange offset along +z).
_HAND_LUMP_MASS: float = 1.66  # 0.73 (panda_hand) + 0.93 (Robotiq 2F-85)
_HAND_LUMP_COM: tuple[float, float, float] = (0.0, 0.0, 0.06)  # rough centroid above flange
_HAND_LUMP_INERTIA: tuple[float, ...] = (0.005, 0.005, 0.0015, 0.0, 0.0, 0.0)

# Franka joint position limits (rad). Used to clip the rolled-out trajectory.
JOINT_POS_MIN: jnp.ndarray = jnp.array(
    [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973], dtype=jnp.float32
)
JOINT_POS_MAX: jnp.ndarray = jnp.array(
    [ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973], dtype=jnp.float32
)


def _inertia_matrix(diag_off: tuple[float, ...]) -> jnp.ndarray:
    ixx, iyy, izz, ixy, ixz, iyz = diag_off
    return jnp.array(
        [[ixx, ixy, ixz],
         [ixy, iyy, iyz],
         [ixz, iyz, izz]],
        dtype=jnp.float32,
    )


def _franka_chain(joint_pos_7: jnp.ndarray, base_world_T: jnp.ndarray):
    """Walk the modified-DH chain.

    Returns:
        T_world: (8, 4, 4) — T_world[0] is the base, T_world[i] for i=1..7 is
                 panda_link{i} in world frame.
        T_pre:   (7, 4, 4) — frame for joint k (k=1..7) BEFORE its rotation is
                 applied. Joint axis k is column 2 of T_pre[k-1].
    """
    T_world_list = [base_world_T]
    T_pre_list = []
    T = base_world_T
    for i, (a, alpha, d, theta_off) in enumerate(_FRANKA_MDH):
        # ^{i-1}T_i(q_i) = T_post(constants) @ Rz(q_i + theta_off).
        T_post = _mdh_transform(a, alpha, d, 0.0)
        T_pre = T @ T_post
        T_pre_list.append(T_pre)
        Rz = _mdh_transform(0.0, 0.0, 0.0, joint_pos_7[i] + theta_off)
        T = T_pre @ Rz
        T_world_list.append(T)
    return jnp.stack(T_world_list), jnp.stack(T_pre_list)


def _body_jacobians(joint_pos_7: jnp.ndarray, base_world_T: jnp.ndarray):
    """Per-body geometric Jacobians and world-frame inertias.

    Bodies 0..6 are panda_link1..panda_link7; body 7 is the hand_lump rigidly
    attached at panda_link8.
    """
    T_world, T_pre = _franka_chain(joint_pos_7, base_world_T)
    z_axes = T_pre[:, :3, 2]    # (7, 3) joint axes in world
    origins = T_pre[:, :3, 3]   # (7, 3) joint origins in world

    flange_T = jnp.eye(4, dtype=base_world_T.dtype).at[2, 3].set(_FLANGE_OFFSET_Z)

    body_T_list = []
    body_com_local_list = []
    body_inertia_local_list = []
    body_mass_list = []
    body_max_joint_list = []  # how many joints in chain affect this body

    for i, (m, com, inertia) in enumerate(_PANDA_LINK_INERTIAS):
        body_T_list.append(T_world[i + 1])
        body_com_local_list.append(jnp.asarray(com, dtype=jnp.float32))
        body_inertia_local_list.append(_inertia_matrix(inertia))
        body_mass_list.append(m)
        body_max_joint_list.append(i + 1)

    body_T_list.append(T_world[7] @ flange_T)  # hand lump at panda_link8
    body_com_local_list.append(jnp.asarray(_HAND_LUMP_COM, dtype=jnp.float32))
    body_inertia_local_list.append(_inertia_matrix(_HAND_LUMP_INERTIA))
    body_mass_list.append(_HAND_LUMP_MASS)
    body_max_joint_list.append(7)

    body_T = jnp.stack(body_T_list)                          # (8, 4, 4)
    body_com_local = jnp.stack(body_com_local_list)          # (8, 3)
    body_inertia_local = jnp.stack(body_inertia_local_list)  # (8, 3, 3)
    body_mass = jnp.array(body_mass_list, dtype=jnp.float32) # (8,)

    R_body = body_T[:, :3, :3]
    p_body = body_T[:, :3, 3]
    com_world = jnp.einsum("bij,bj->bi", R_body, body_com_local) + p_body          # (8, 3)
    I_world = jnp.einsum("bij,bjk,blk->bil", R_body, body_inertia_local, R_body)   # (8, 3, 3)

    # Mask (8, 7): does joint k (0-indexed) affect body i?
    k_range = jnp.arange(7)
    max_joint_arr = jnp.asarray(body_max_joint_list)[:, None]
    mask = (k_range[None, :] < max_joint_arr).astype(jnp.float32)  # (8, 7)

    # Linear/angular Jacobians.
    # J_v[i, :, k] = z_k × (com_i - origin_k); J_w[i, :, k] = z_k.
    diff = com_world[:, None, :] - origins[None, :, :]               # (8, 7, 3)
    z_b = jnp.broadcast_to(z_axes[None, :, :], (8, 7, 3))            # (8, 7, 3)
    Jv_full = jnp.cross(z_b, diff)                                   # (8, 7, 3)
    Jv = jnp.einsum("bk,bkj->bjk", mask, Jv_full)                    # (8, 3, 7)
    Jw = jnp.einsum("bk,bkj->bjk", mask, z_b)                        # (8, 3, 7)

    return Jv, Jw, body_mass, I_world


def franka_mass_matrix(joint_pos_7: jnp.ndarray, base_world_T: jnp.ndarray) -> jnp.ndarray:
    """Configuration-dependent 7×7 mass matrix M(q)."""
    Jv, Jw, masses, I_world = _body_jacobians(joint_pos_7, base_world_T)
    M_v = jnp.einsum("b,bki,bkj->ij", masses, Jv, Jv)
    M_w = jnp.einsum("bki,bkl,blj->ij", Jw, I_world, Jw)
    return M_v + M_w


def implicit_pd_substep(
    q: jnp.ndarray,
    v: jnp.ndarray,
    q_cmd: jnp.ndarray,
    *,
    base_world_T: jnp.ndarray,
    dt: jnp.ndarray,
    K: jnp.ndarray,
    D: jnp.ndarray,
    q_min: jnp.ndarray,
    q_max: jnp.ndarray,
):
    """One physics substep. Returns ``(q_new, v_new)``."""
    M = franka_mass_matrix(q, base_world_T)
    eye7 = jnp.eye(7, dtype=q.dtype)
    A = M + (dt * D) * eye7 + (dt * dt * K) * eye7
    b = M @ v + (dt * K) * (q_cmd - q)
    v_new = jnp.linalg.solve(A, b)
    q_new = q + dt * v_new
    q_clipped = jnp.clip(q_new, q_min, q_max)
    at_lower = q_new <= q_min
    at_upper = q_new >= q_max
    v_new = jnp.where(at_lower & (v_new < 0), 0.0, v_new)
    v_new = jnp.where(at_upper & (v_new > 0), 0.0, v_new)
    return q_clipped, v_new


def actuator_rollout_single(
    q0: jnp.ndarray,
    v0: jnp.ndarray,
    q_cmd_chunk: jnp.ndarray,
    *,
    base_world_T: jnp.ndarray,
    dt: float,
    decimation: int,
    K: float,
    D: float,
    q_min: jnp.ndarray = JOINT_POS_MIN,
    q_max: jnp.ndarray = JOINT_POS_MAX,
) -> jnp.ndarray:
    """Roll forward H control steps × ``decimation`` physics substeps.

    Args:
        q0, v0: (7,) initial joint position and velocity.
        q_cmd_chunk: (H, 7) commanded joint positions, one per control step.
    Returns:
        q_chunk: (H, 7) joint positions at the end of each control step.
    """
    dt_j = jnp.asarray(dt, dtype=q0.dtype)
    K_j = jnp.asarray(K, dtype=q0.dtype)
    D_j = jnp.asarray(D, dtype=q0.dtype)

    def _control_step(carry, q_cmd_h):
        def _substep(state, _):
            q, v = state
            q, v = implicit_pd_substep(
                q, v, q_cmd_h,
                base_world_T=base_world_T,
                dt=dt_j, K=K_j, D=D_j,
                q_min=q_min, q_max=q_max,
            )
            return (q, v), None
        carry, _ = jax.lax.scan(_substep, carry, jnp.zeros(decimation, dtype=q0.dtype))
        return carry, carry[0]  # report q after this control step

    _, q_chunk = jax.lax.scan(_control_step, (q0, v0), q_cmd_chunk)
    return q_chunk


def actuator_rollout(
    q0: jnp.ndarray,
    v0: jnp.ndarray,
    q_cmd_chunk: jnp.ndarray,
    *,
    base_world_T: jnp.ndarray,
    dt: float,
    decimation: int,
    K: float,
    D: float,
    q_min: jnp.ndarray | None = None,
    q_max: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Batched rollout. Shapes: q0 (B, 7), v0 (B, 7), q_cmd_chunk (B, H, 7)."""
    if q_min is None:
        q_min = JOINT_POS_MIN
    if q_max is None:
        q_max = JOINT_POS_MAX

    def _wrapped(q0_, v0_, q_cmd_):
        return actuator_rollout_single(
            q0_, v0_, q_cmd_,
            base_world_T=base_world_T,
            dt=dt, decimation=decimation, K=K, D=D,
            q_min=q_min, q_max=q_max,
        )

    return jax.vmap(_wrapped)(q0, v0, q_cmd_chunk)
