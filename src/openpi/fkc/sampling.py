"""Guided sampling loops: vanilla / linear_combo / fkc.

This is the user-designated "main inference loops" file — the three modes live
here so the cost/constraint and kinematics helpers can stay in their own small
modules.

Flow matching → SDE conversion
------------------------------
pi0.5 trains a flow matching velocity ``v_t`` where ``v_t`` IS the
probability-flow-ODE drift. To apply FKC (which is derived for a reverse SDE)
we lift the ODE into an equivalent-marginals SDE per "Stochastic Sampling from
Deterministic Flow Models". For a flow with convention
``x_t = t·noise + (1-t)·actions`` and ``v = noise - actions`` (pi0.5's
convention, t: 1 → 0):

    score(x_t) = ∇log p_t(x_t) = -(x_t + (1-t)·v_t) / t     (Tweedie)
    SDE drift   = v_t + ½σ²·score                          (marginal-preserving)
    diffusion   = σ_t dW_t

Add Gibbs-tilt guidance (FKC-PDE note, eq 22, translated to flow matching):

    drift_guided = v_t + ½σ²·(score - β_t·∇J_grad)
    Δlog_w       = β_t·⟨∇J_grad, v_t⟩·dt + (β_{t+dt} - β_t)·J_value

Sign note: gr00t's formula has ``+ β_t·∇J`` in the drift because their sampling
time runs 0 → 1 with ``dt > 0`` and ``β < 0`` means "pull toward low cost" — the
integration step ``dt·β·∇J`` then points along ``−∇J``. pi0.5 samples the other
way (t: 1 → 0, ``dt < 0``), so the same ``+β·∇J`` term would flip to ``+∇J``
under integration and push toward *high* cost. We keep ``β < 0`` as the
"pull toward low cost" convention (so the log-weight formula is unchanged —
``β·⟨∇J,v⟩·dt`` picks up two sign flips in pi0.5 that cancel) and instead
subtract ``β_t·∇J_grad`` in the drift to restore the correct direction.

Three modes, all sharing the same step primitive:

    vanilla:      x_{t+dt} = x_t + dt·v_t                               (deterministic flow)
    linear_combo: x_{t+dt} = x_t + dt·drift_guided + σ√|dt|·ε           (SDE, 1 particle)
    fkc:          same drift + log-weight update + resampling            (SDE, K particles)

Decoupled weights (the user-requested change from gr00t):
    - J_value  uses the "value"   weights (``w_*_value``) and enters the log-weight jump.
    - ∇J_grad  uses the "gradient" weights (``w_*_grad``)  and enters the drift.
    Setting the two weight sets equal recovers gr00t-style FKC.

Note on σ=0: for linear_combo and fkc modes, the entire gradient contribution
is scaled by σ²/2, so σ_schedule=``zero`` makes guidance vanish. To get
meaningful guidance, set ``sigma_schedule`` to ``zero_ends``, ``constant``, or
``non_singular`` (Singh & Fischer Table 1) with ``sigma_scale > 0`` in the YAML.
"""

from __future__ import annotations

from typing import Any

import einops
import jax
import jax.numpy as jnp

from openpi.fkc import cost_constraint as _cc
from openpi.fkc.config import FKCConfig
from openpi.models import model as _model


# ---------------------------------------------------------------------------
# Schedules
# ---------------------------------------------------------------------------


def beta_value(time: jnp.ndarray, fkc_config: FKCConfig) -> jnp.ndarray:
    """β_t for the Feynman-Kac weighting. Signed to pull toward low cost.

    ``time`` follows openpi's convention: 1.0 at pure noise, 0.0 at clean data.
    """
    strength = jnp.asarray(fkc_config.beta_strength, dtype=jnp.float32)
    if fkc_config.beta_schedule == "constant":
        gamma = strength
    elif fkc_config.beta_schedule == "linear_ramp":
        # Stronger near clean-data end.
        gamma = strength * (1.0 - time)
    else:  # defensive — the dataclass Literal already restricts values.
        raise ValueError(f"unknown beta_schedule: {fkc_config.beta_schedule}")
    return -gamma


def sigma_value(time: jnp.ndarray, fkc_config: FKCConfig) -> jnp.ndarray:
    """Diffusion coefficient σ_t used for the stochastic corrector step."""
    scale = jnp.asarray(fkc_config.sigma_scale, dtype=jnp.float32)
    if fkc_config.sigma_schedule == "zero":
        return jnp.asarray(0.0, dtype=jnp.float32)
    if fkc_config.sigma_schedule == "constant":
        return scale
    if fkc_config.sigma_schedule == "zero_ends":
        return scale * jnp.sqrt(jnp.clip(time * (1.0 - time), 0.0, None))
    if fkc_config.sigma_schedule == "non_singular":
        # NonSingular from Singh & Fischer 2024 Table 1: σ_t = α·√t in their
        # t-convention where t=1 is the source noise. pi0.5 shares that
        # convention, so the schedule maps over directly (peaks at pure noise,
        # zero at clean data). This is the mirror of gr00t's α·√(1 - t_cont).
        return scale * jnp.sqrt(jnp.clip(time, 0.0, None))
    raise ValueError(f"unknown sigma_schedule: {fkc_config.sigma_schedule}")


def score_from_velocity(
    x_t: jnp.ndarray,
    v_t: jnp.ndarray,
    time: jnp.ndarray,
    *,
    source_mean: float | jnp.ndarray = 0.0,
    source_variance: float | jnp.ndarray = 1.0,
    eps: float = 1.0e-6,
) -> jnp.ndarray:
    """Score ∇log p_t(x_t) derived from the flow-matching velocity.

    pi0.5 convention: ``x_t = t·noise + (1-t)·actions`` with ``v = noise - actions``
    and ``t: 1 → 0``. Parametrising the prior p_{t=1} as N(μ, σ² I), Singh &
    Fischer 2024 Eq. 13 gives

        ∇log p_t(x_t) = (μ - x_t - (1-t)·v_t) / (t · σ²)

    The defaults ``μ=0, σ²=1`` recover pi0.5's standard-normal prior and the
    familiar ``-(x_t + (1-t)·v_t) / t``.

    Sanity: at t=1 (pure prior), score = (μ - x_t)/σ², which is ∇log N(μ, σ²I). ✓
    Diverges at t=0 (clean data); the sampling loop stops before reaching t=0.

    This is the openpi-time-convention mirror of gr00t's
    ``_score_from_velocity`` (which uses (t·v + μ - x)/((1-t)·σ²) for its
    flipped time convention).
    """
    denom = jnp.maximum(time, eps) * source_variance
    return (source_mean - x_t - (1.0 - time) * v_t) / denom


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------


def _warn_invalid(count: jnp.ndarray, where: str) -> None:
    """Emit a one-line warning when a batch row's weights are degenerate.

    ``jax.lax.cond`` + ``jax.debug.print`` keeps this JIT-safe: the print
    callback fires on the host only when ``count > 0``.
    """
    def _emit(c):
        jax.debug.print(
            "[FKC] {where}: {c} batch item(s) had degenerate weights; falling back to uniform.",
            where=where,
            c=c,
        )

    def _noop(_c):
        return None

    jax.lax.cond(count > 0, _emit, _noop, count)


def _systematic_resample(
    log_weights: jnp.ndarray,
    rng: jax.Array,
    handle_invalid: bool,
) -> jnp.ndarray:
    """Systematic (low-variance) resampling. Returns (B, K) int32 indices.

    When ``handle_invalid`` is True, batch rows whose post-softmax weights are
    non-finite or sum to zero are replaced by a uniform distribution and a
    warning is printed (matches gr00t's robustness fallback). When False, bad
    weights propagate silently.
    """
    b, k = log_weights.shape
    shifted = log_weights - jnp.max(log_weights, axis=1, keepdims=True)
    weights = jax.nn.softmax(shifted, axis=1)

    if handle_invalid:
        invalid = jnp.any(~jnp.isfinite(weights), axis=1) | (jnp.sum(weights, axis=1) <= 0)
        _warn_invalid(jnp.sum(invalid.astype(jnp.int32)), where="systematic_resample")
        uniform = jnp.full_like(weights, 1.0 / k)
        weights = jnp.where(invalid[:, None], uniform, weights)

    cdf = jnp.cumsum(weights, axis=1)
    cdf = cdf.at[:, -1].set(1.0)

    base = jax.random.uniform(rng, (b, 1), dtype=log_weights.dtype) / k
    steps = jnp.arange(k, dtype=log_weights.dtype)
    thresholds = jnp.clip(base + steps[None, :] / k, 0.0, 1.0 - 1.0e-7)
    return jnp.clip(
        jax.vmap(jnp.searchsorted)(cdf, thresholds),
        a_max=k - 1,
    ).astype(jnp.int32)


def _gather_particles(actions: jnp.ndarray, indices: jnp.ndarray, batch_size: int, num_particles: int) -> jnp.ndarray:
    """Gather per-batch particles using (B, K) indices.

    ``actions`` is (B*K, H, A). Result has the same shape but with the particle
    axis reshuffled according to ``indices``.
    """
    b, k = batch_size, num_particles
    traj = actions.reshape(b, k, *actions.shape[1:])  # (B, K, H, A)
    gathered = jnp.take_along_axis(traj, indices[..., None, None], axis=1)
    return gathered.reshape(b * k, *actions.shape[1:])


def _sample_final_particle(
    log_weights: jnp.ndarray,
    rng: jax.Array,
    handle_invalid: bool,
) -> jnp.ndarray:
    """Draw one particle per batch item by multinomial over final weights.

    When ``handle_invalid`` is True, rows with non-finite log-weights are
    replaced by zero-logits (uniform) and a warning is printed.
    """
    if handle_invalid:
        invalid = jnp.any(~jnp.isfinite(log_weights), axis=1)
        _warn_invalid(jnp.sum(invalid.astype(jnp.int32)), where="sample_final_particle")
        safe_logw = jnp.where(invalid[:, None], jnp.zeros_like(log_weights), log_weights)
        shifted = safe_logw - jnp.max(safe_logw, axis=1, keepdims=True)
    else:
        shifted = log_weights - jnp.max(log_weights, axis=1, keepdims=True)
    return jax.random.categorical(rng, shifted, axis=1).astype(jnp.int32)


# ---------------------------------------------------------------------------
# Observation / particle tiling
# ---------------------------------------------------------------------------


def _tile_for_particles(obs: _model.Observation, num_particles: int) -> _model.Observation:
    """Repeat every leaf of ``obs`` along axis 0 by ``num_particles``."""
    if num_particles == 1:
        return obs

    def _tile(x: Any) -> Any:
        return jnp.repeat(x, num_particles, axis=0) if isinstance(x, jnp.ndarray) else x

    return _model.Observation(
        images={k: _tile(v) for k, v in obs.images.items()},
        image_masks={k: _tile(v) for k, v in obs.image_masks.items()},
        state=_tile(obs.state),
        tokenized_prompt=_tile(obs.tokenized_prompt) if obs.tokenized_prompt is not None else None,
        tokenized_prompt_mask=_tile(obs.tokenized_prompt_mask) if obs.tokenized_prompt_mask is not None else None,
        token_ar_mask=_tile(obs.token_ar_mask) if obs.token_ar_mask is not None else None,
        token_loss_mask=_tile(obs.token_loss_mask) if obs.token_loss_mask is not None else None,
    )


# ---------------------------------------------------------------------------
# KV cache + velocity helpers (lean wrappers around Pi0 internals)
# ---------------------------------------------------------------------------


def _build_prefix_cache(model: Any, obs: _model.Observation):
    """Run the prefix through the LLM once to populate a KV cache.

    Duplicates the setup in Pi0.sample_actions so the guided loop can re-use
    the same cache across all denoising steps.
    """
    from openpi.models.pi0 import make_attn_mask  # local import to avoid cycle

    prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(obs)
    prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
    positions = jnp.cumsum(prefix_mask, axis=1) - 1
    _, kv_cache = model.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)
    return kv_cache, prefix_mask, prefix_tokens.shape[1]


def _predict_velocity(
    model: Any,
    obs: _model.Observation,
    x_t: jnp.ndarray,
    time: jnp.ndarray,
    kv_cache: Any,
    prefix_mask: jnp.ndarray,
    prefix_token_count: int,
) -> jnp.ndarray:
    """Compute v_t(x_t) using the pre-built KV cache. Mirrors Pi0.sample_actions.step."""
    from openpi.models.pi0 import make_attn_mask

    batch_size = x_t.shape[0]
    suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = model.embed_suffix(
        obs, x_t, jnp.broadcast_to(time, batch_size)
    )
    suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
    prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
    full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
    assert full_attn_mask.shape == (batch_size, suffix_tokens.shape[1], prefix_token_count + suffix_tokens.shape[1])
    positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
    (prefix_out, suffix_out), _ = model.PaliGemma.llm(
        [None, suffix_tokens],
        mask=full_attn_mask,
        positions=positions,
        kv_cache=kv_cache,
        adarms_cond=[None, adarms_cond],
    )
    assert prefix_out is None
    return model.action_out_proj(suffix_out[:, -model.action_horizon :])


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------


def sample_actions_guided(
    model: Any,
    rng: jax.Array,
    observation: _model.Observation,
    fkc_config: FKCConfig,
    fkc_runtime: _cc.FKRuntime,
    *,
    num_steps: int,
    noise: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Run vanilla / linear_combo / fkc sampling.

    ``fkc_config`` must be static for jit caching (frozen dataclass -> hashable).
    ``fkc_runtime`` is a PyTree of jnp arrays, traced normally.
    """
    observation = _model.preprocess_observation(None, observation, train=False)
    dt = -1.0 / num_steps
    mode = fkc_config.mode

    orig_batch_size = observation.state.shape[0]
    num_particles = max(1, fkc_config.num_particles) if mode == "fkc" else 1
    total_batch = orig_batch_size * num_particles

    # Tile obs + runtime across particles (no-op when num_particles == 1).
    obs_tiled = _tile_for_particles(observation, num_particles)
    rt_tiled = _cc._repeat_runtime_for_particles(fkc_runtime, num_particles)

    # Draw / reshape noise.
    if noise is None:
        rng, noise_rng = jax.random.split(rng)
        noise = jax.random.normal(noise_rng, (total_batch, model.action_horizon, model.action_dim))
    elif num_particles > 1:
        noise = jnp.repeat(noise, num_particles, axis=0)

    kv_cache, prefix_mask, prefix_token_count = _build_prefix_cache(model, obs_tiled)

    # Unrolled python loop — num_steps is static, JIT handles it.
    x_t = noise
    time = jnp.asarray(1.0, dtype=jnp.float32)
    log_weights = jnp.zeros((orig_batch_size, num_particles), dtype=jnp.float32)

    sqrt_abs_dt = jnp.sqrt(jnp.asarray(abs(dt), dtype=jnp.float32))

    for step_idx in range(num_steps):
        rng, noise_rng, resample_rng = jax.random.split(rng, 3)
        t_cur = 1.0 + dt * step_idx
        t_next = 1.0 + dt * (step_idx + 1)
        t_cur_j = jnp.asarray(t_cur, dtype=jnp.float32)
        t_next_j = jnp.asarray(t_next, dtype=jnp.float32)

        v_t = _predict_velocity(model, obs_tiled, x_t, t_cur_j, kv_cache, prefix_mask, prefix_token_count)

        if mode == "vanilla":
            x_t = x_t + dt * v_t
            time = t_next_j
            continue

        # linear_combo / fkc: convert flow to SDE via score compensation, then
        # add the Gibbs-tilt gradient drift. See module docstring for derivation.
        j_val, grad_j = _cc.J_value_and_grad(x_t, rt_tiled, fkc_config)

        # Force serial NN -> cost scheduling when the user requests it. The jit
        # compiler otherwise overlaps them on GPU where possible.
        if not fkc_config.parallel_nn_and_cost:
            grad_j = grad_j + 0.0 * jnp.sum(v_t)

        score = score_from_velocity(
            x_t,
            v_t,
            t_cur_j,
            source_mean=fkc_config.flow_source_mean,
            source_variance=fkc_config.flow_source_variance,
        )
        sigma = sigma_value(t_cur_j, fkc_config)
        beta_t = beta_value(t_cur_j, fkc_config)

        # FKC-PDE eq 22 adapted to pi0.5's flow-matching velocity. pi0.5 runs
        # the integrator with dt < 0 (t: 1 → 0), which flips the sign of the
        # guidance term relative to gr00t's dt > 0 convention — so we subtract
        # β_t·∇J_grad here even though gr00t adds it. See module docstring.
        # When σ=0 the cost contribution and score compensation both vanish,
        # i.e. guidance is inactive — users must set sigma_scale > 0 for
        # linear_combo / fkc modes to have any effect.
        half_sigma_sq = 0.5 * sigma * sigma
        drift = v_t + half_sigma_sq * (score - beta_t * grad_j)

        if fkc_config.sigma_schedule != "zero":
            eps = jax.random.normal(noise_rng, x_t.shape, dtype=x_t.dtype)
            x_t = x_t + dt * drift + sigma * sqrt_abs_dt * eps
        else:
            x_t = x_t + dt * drift

        if mode == "fkc":
            in_window = (fkc_config.resample_t_min <= t_cur <= fkc_config.resample_t_max)
            if in_window:
                beta_next = beta_value(t_next_j, fkc_config)
                # ⟨∇J_grad, v_t⟩ — v_t IS the probability-flow ODE drift, so this
                # matches the ⟨∇J, σ²/2·∇log q - f⟩ term in eq 23 of the notes.
                inner = jnp.sum(grad_j * v_t, axis=(-2, -1))  # (B*K,)
                dlogw = beta_t * inner * dt + (beta_next - beta_t) * j_val
                dlogw = dlogw.reshape(orig_batch_size, num_particles).astype(jnp.float32)
                if fkc_config.center_log_weight_increment:
                    dlogw = dlogw - jnp.mean(dlogw, axis=1, keepdims=True)
                log_weights = log_weights + dlogw

                should_resample = (
                    num_particles > 1
                    and (step_idx + 1) % fkc_config.resample_interval == 0
                    and (step_idx + 1) < num_steps
                )
                if should_resample:
                    indices = _systematic_resample(
                        log_weights, resample_rng, fkc_config.handle_invalid_weights
                    )
                    x_t = _gather_particles(x_t, indices, orig_batch_size, num_particles)
                    log_weights = jnp.zeros_like(log_weights)

        time = t_next_j

    if mode == "fkc" and num_particles > 1:
        rng, pick_rng = jax.random.split(rng)
        picked = _sample_final_particle(
            log_weights, pick_rng, fkc_config.handle_invalid_weights
        )  # (B,)
        x_t = x_t.reshape(orig_batch_size, num_particles, *x_t.shape[1:])
        x_t = jnp.take_along_axis(x_t, picked[:, None, None, None], axis=1).squeeze(1)

    return x_t
