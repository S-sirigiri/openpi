"""FKC guidance configuration.

Single-source-of-truth for the "which guidance mode and with what knobs" question
feeding ``openpi.fkc.sampling``. Configs are authored as YAML and read into the
frozen ``FKCConfig`` dataclass below.

Key user-requested spec: objective weights for the *value* of the cost/constraint
terms (``w_*_value``) are decoupled from the weights for the *gradient* term
(``w_*_grad``). That gives the caller independent control over how strongly a
term contributes to (a) the Feynman-Kac log-weight update and (b) the augmented
drift applied to x_t at each denoising step.
"""

from __future__ import annotations

import dataclasses
import pathlib
from typing import Any, Literal

import yaml


@dataclasses.dataclass(frozen=True)
class FKConfig:
    """Forward-kinematics configuration."""

    # Name of the robot kinematic chain to use. Only "franka_panda" is implemented today.
    robot: Literal["franka_panda"] = "franka_panda"

    # Robot base pose in world frame, as (xyz, xyzw-quat). Franka's base is at the
    # world origin in RoboLab, so the identity default is correct for
    # BananaInBowlTask.
    base_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    base_quat_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)

    # Extra fixed transform from the Franka flange to the body whose world pose we
    # want to match. RoboLab treats the Robotiq 2F-85 ``base_link`` as the EE body
    # (see RoboLab observations.py) and the flattened USD merges the mount rigidly
    # into panda_link8, so we keep this as identity by default and fine-tune it
    # with the online FK-vs-ee_pos comparison.
    ee_offset_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    ee_offset_quat_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)


@dataclasses.dataclass(frozen=True)
class DynamicsConfig:
    """Mini-simulator config for ``openpi.fkc.dynamics`` actuator rollout.

    Defaults match RoboLab's ``DroidCfg`` + ``BananaInBowlTask`` env (Isaac Lab
    ``ImplicitActuator`` with K=400, D=80; sim dt=1/120 s; decimation=8).
    Disable (``enabled: false``) to fall back to the original "perfect
    tracking" proxy.
    """

    enabled: bool = True

    # PD gains applied to all 7 arm joints. Match the actuator config in the
    # target simulator. Wrong values here = wrong trajectory prediction.
    stiffness: float = 400.0
    damping: float = 80.0

    # Physics dt (seconds) and decimation (substeps per control step). Both
    # come straight from the env config (env_cfg.sim.dt, env_cfg.decimation).
    sim_dt: float = 1.0 / 120.0
    decimation: int = 8


@dataclasses.dataclass(frozen=True)
class CostConfig:
    """Placeholder cost / constraint parameters.

    The real task cost (banana-to-bowl etc.) will replace these later. For now
    the shipping default drives the EE toward ``target_xyz`` in world frame and
    keeps it inside a large bounding box so the constraint path is exercised
    end-to-end without affecting rollouts much.
    """

    # A static target in world frame that the final EE position is pulled toward.
    target_xyz: tuple[float, float, float] = (0.55, 0.0, 0.25)

    # Inequality constraint: axis-aligned world-frame bounding box that every EE
    # waypoint must stay inside. Wide enough by default to be mostly inactive.
    box_min_xyz: tuple[float, float, float] = (-1.0, -1.0, 0.0)
    box_max_xyz: tuple[float, float, float] = (1.0, 1.0, 1.2)

    # Softplus sharpness used to smooth the inequality hinge (higher = closer to
    # a hard hinge, matches gr00t's ``softplus_beta`` knob).
    softplus_beta: float = 10.0


@dataclasses.dataclass(frozen=True)
class FKCConfig:
    """Top-level FKC guidance config.

    Mode selection:
        - ``vanilla``: call the original pi0.5 sample_actions unchanged.
        - ``linear_combo``: single particle, drift augmented by cost gradient.
        - ``fkc``: N particles in parallel, per-step Feynman-Kac log-weight
          updates, optional systematic resampling, final multinomial pick.
    """

    mode: Literal["vanilla", "linear_combo", "fkc"] = "vanilla"

    # Integrator step count. ``None`` means "inherit from model.sample_actions"
    # (openpi default is 10 for pi0.5).
    num_steps: int | None = None

    # --- Particles -----------------------------------------------------------
    num_particles: int = 1
    # If True, expand the batch dim by ``num_particles`` and run a single model
    # forward per step (fully parallel particles). If False, run K sequential
    # forwards (useful for memory-constrained debugging).
    particles_parallel: bool = True

    # --- Compute scheduling --------------------------------------------------
    # Best-effort knob. When True, the NN forward and the cost/grad are both
    # launched inside the same jit'd step — XLA typically overlaps them on GPU.
    # When False, we sequence them explicitly (cost_and_grad starts only after
    # the model forward is materialised) for benchmarking/debugging.
    parallel_nn_and_cost: bool = True

    # --- Decoupled objective weights -----------------------------------------
    # "value" weights shape J(x) that goes into the Feynman-Kac log-weight jump.
    w_cost_value: float = 1.0
    w_eq_value: float = 1.0
    w_ineq_value: float = 1.0
    # "grad" weights shape the augmented drift ∇J_grad(x) applied to x_t.
    w_cost_grad: float = 1.0
    w_eq_grad: float = 1.0
    w_ineq_grad: float = 1.0

    # --- Gibbs-tilt (β_t) schedule -------------------------------------------
    # Signed: negative β pulls toward low-cost (the usual setting in the notes).
    # ``strength`` sets |β| and ``schedule`` shapes its time dependence.
    beta_schedule: Literal["constant", "linear_ramp"] = "constant"
    beta_strength: float = 1.0

    # --- Stochasticity (flow → SDE, cf. Stochastic Sampling paper) -----------
    # ``zero`` keeps the integrator deterministic, matching vanilla pi0.5.
    # ``constant`` / ``zero_ends`` / ``non_singular`` inject Brownian noise at
    # every step, letting FKC re-sample around the posterior mode rather than
    # collapsing to a single point. ``non_singular`` is Table 1 of Singh &
    # Fischer 2024 (σ_t = α·√t in pi0's t=1-is-noise convention) and was their
    # best-performing ImageNet schedule — high noise at the noise end, zero at
    # the data end.
    sigma_schedule: Literal["zero", "constant", "zero_ends", "non_singular"] = "zero"
    sigma_scale: float = 0.0

    # --- Flow-source (prior) distribution ------------------------------------
    # Parameters (mean, variance) of p_{t=1} — the prior that x_t converges to
    # at pure noise. pi0.5 uses N(0, I), so the defaults keep the sampler
    # identical to unguided pi0.5 when FKC is off. Plugged into Singh & Fischer
    # 2024 Eq. 13 to derive the score from the flow velocity.
    flow_source_mean: float = 0.0
    flow_source_variance: float = 1.0

    # --- Resampling (fkc mode only) ------------------------------------------
    resample_interval: int = 1
    resample_t_min: float = 0.05
    resample_t_max: float = 0.95
    center_log_weight_increment: bool = True
    # When True, batch rows with non-finite / zero-sum log-weights are replaced
    # by a uniform distribution in ``_systematic_resample`` and
    # ``_sample_final_particle``, and a warning is printed. Guards against
    # numerical blow-up when β is large or J explodes. Turn off to get raw
    # (potentially NaN) behavior.
    handle_invalid_weights: bool = True

    # --- Sub-configs ---------------------------------------------------------
    fk: FKConfig = dataclasses.field(default_factory=FKConfig)
    cost: CostConfig = dataclasses.field(default_factory=CostConfig)
    dynamics: DynamicsConfig = dataclasses.field(default_factory=DynamicsConfig)


def _coerce_tuple(value: Any, length: int) -> tuple[float, ...]:
    if value is None:
        raise ValueError("expected a list of floats, got None")
    out = tuple(float(v) for v in value)
    if len(out) != length:
        raise ValueError(f"expected length-{length} tuple, got {out}")
    return out


def load_fkc_config(path: str | pathlib.Path) -> FKCConfig:
    """Read ``path`` (YAML) into an ``FKCConfig``.

    Unknown keys raise ``TypeError`` so typos don't silently vanish.
    """
    p = pathlib.Path(path)
    with p.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"expected a mapping in {p}, got {type(raw).__name__}")

    fk_raw = dict(raw.pop("fk", {}) or {})
    cost_raw = dict(raw.pop("cost", {}) or {})
    dyn_raw = dict(raw.pop("dynamics", {}) or {})

    for key in ("base_xyz", "ee_offset_xyz"):
        if key in fk_raw:
            fk_raw[key] = _coerce_tuple(fk_raw[key], 3)
    for key in ("base_quat_xyzw", "ee_offset_quat_xyzw"):
        if key in fk_raw:
            fk_raw[key] = _coerce_tuple(fk_raw[key], 4)
    fk = FKConfig(**fk_raw)

    for key in ("target_xyz", "box_min_xyz", "box_max_xyz"):
        if key in cost_raw:
            cost_raw[key] = _coerce_tuple(cost_raw[key], 3)
    cost = CostConfig(**cost_raw)

    dynamics = DynamicsConfig(**dyn_raw)

    return FKCConfig(fk=fk, cost=cost, dynamics=dynamics, **raw)
