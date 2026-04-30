"""Turn a chunk of model actions into a world-frame EE path.

The model outputs normalized delta joint positions (action_dim padded to 32;
only the first 7 dims are the Franka arm deltas). At inference the final
``Unnormalize`` + ``AbsoluteActions`` transforms convert those back to absolute
world-space joint commands.

FKC guidance fires *inside* the sampling loop — before any of those output
transforms run — so we replicate their joint-only subset here so the cost layer
can reason about world coordinates.

Two proxy modes for the chunk's joint trajectory:

* ``dynamics_cfg.enabled = True`` (default) — roll the chunk through a JAX
  copy of Isaac Lab's ``ImplicitActuator`` PD law (see ``dynamics.py``). This
  is the principled trajectory the simulator will *actually* execute, including
  controller lag and joint-position-limit clipping.
* ``dynamics_cfg.enabled = False`` — the original "perfect tracking" proxy:
  the rolled-out joint state at horizon h equals the commanded action at h.
  Kept for ablation/debugging; the constraint will be optimistic about the
  controller's tracking ability.

Inputs the cost uses:
    - actions: (B, horizon, action_dim) still in the normalized delta space
    - current_joint_pos: (B, 7) absolute, in radians, the proprio state
    - current_joint_vel: (B, 7) absolute, in rad/s — initial v for the rollout
    - norm stats for the first 7 action dims
Output:
    - world_path: (B, horizon, 3) EE positions along the predicted horizon
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from openpi.fkc.dynamics import actuator_rollout
from openpi.fkc.fk import franka_fk_position


def unnormalize_joint_deltas(
    normalized_deltas: jnp.ndarray,
    action_mean_7: jnp.ndarray,
    action_std_7: jnp.ndarray,
) -> jnp.ndarray:
    """Undo openpi's z-score normalization on the first 7 action dims."""
    return normalized_deltas * (action_std_7 + 1e-6) + action_mean_7


def build_world_path(
    actions: jnp.ndarray,
    current_joint_pos: jnp.ndarray,
    current_joint_vel: jnp.ndarray,
    action_mean_7: jnp.ndarray,
    action_std_7: jnp.ndarray,
    base_world_T: jnp.ndarray,
    ee_offset_T: jnp.ndarray,
    dynamics_cfg: Any,
) -> jnp.ndarray:
    """Convert a chunk of normalized delta actions to a world EE path.

    Args:
        actions: (B, H, A) normalized action chunk from the sampling loop
            (B may be ``batch * num_particles`` for FKC).
        current_joint_pos: (B, 7) absolute joint positions (radians).
        current_joint_vel: (B, 7) absolute joint velocities (rad/s). Used as
            the initial velocity for the actuator rollout. Pass zeros when the
            simulator-reported velocity is unavailable.
        action_mean_7, action_std_7: (7,) action normalization stats.
        base_world_T, ee_offset_T: (4,4) static SE(3) transforms.
        dynamics_cfg: ``DynamicsConfig`` controlling the actuator rollout.

    Returns:
        (B, H, 3) EE positions in world coordinates.
    """
    arm_deltas_norm = actions[..., :7]
    arm_deltas = unnormalize_joint_deltas(arm_deltas_norm, action_mean_7, action_std_7)
    # Absolute joint commands per horizon step (matches the AbsoluteActions
    # output transform applied to model outputs at inference).
    q_cmd_chunk = arm_deltas + current_joint_pos[..., None, :]  # (B, H, 7)

    if dynamics_cfg.enabled:
        # Roll the chunk through Isaac Lab's PD law. Output: (B, H, 7) — the
        # joint trajectory the simulator will actually execute.
        abs_joints = actuator_rollout(
            current_joint_pos,
            current_joint_vel,
            q_cmd_chunk,
            base_world_T=base_world_T,
            dt=dynamics_cfg.sim_dt,
            decimation=dynamics_cfg.decimation,
            K=dynamics_cfg.stiffness,
            D=dynamics_cfg.damping,
        )
    else:
        # Legacy "perfect tracking" proxy: the chunk's joint trajectory IS the
        # commanded chunk. Useful for ablation only.
        abs_joints = q_cmd_chunk

    fk_pos = jax.vmap(jax.vmap(lambda q: franka_fk_position(q, base_world_T, ee_offset_T)))
    return fk_pos(abs_joints)
