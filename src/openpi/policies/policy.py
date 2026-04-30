from collections.abc import Sequence
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.fkc import cost_constraint as _fkc_cc
from openpi.fkc import fk as _fkc_fk
from openpi.fkc.config import FKCConfig
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
        fkc_config: FKCConfig | None = None,
        norm_stats: dict[str, _transforms.NormStats] | None = None,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
            fkc_config: Optional FKC guidance config. If given and not
                ``mode=="vanilla"``, inference routes through the guided sampler.
                Only supported on the JAX path.
            norm_stats: Normalization statistics. Required alongside ``fkc_config``
                so the guided sampler can un-normalize joint states for FK.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device
        self._fkc_config = fkc_config
        self._fkc_active = fkc_config is not None and fkc_config.mode != "vanilla"

        if self._is_pytorch_model:
            if self._fkc_active:
                raise NotImplementedError("FKC guidance is only implemented for the JAX inference path.")
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
        else:
            # JAX model setup
            if self._fkc_active:
                if norm_stats is None or "state" not in norm_stats or "actions" not in norm_stats:
                    raise ValueError(
                        "FKC guidance requires 'state' and 'actions' entries in norm_stats so "
                        "joint positions can be un-normalized for forward kinematics."
                    )
                self._fkc_static_runtime_parts = self._build_fkc_static_runtime(fkc_config, norm_stats)
                self._resolved_num_steps = int(fkc_config.num_steps) if fkc_config.num_steps is not None else 10
                self._sample_actions = nnx_utils.module_jit(
                    model.sample_actions_guided,
                    static_argnames=("fkc_config", "num_steps"),
                )
            else:
                self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            self._rng = rng or jax.random.key(0)

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        if self._fkc_active:
            sample_kwargs = dict(sample_kwargs)
            sample_kwargs["fkc_config"] = self._fkc_config
            sample_kwargs["fkc_runtime"] = self._materialise_fkc_runtime(observation)
            sample_kwargs["num_steps"] = self._resolved_num_steps
        outputs = {
            "state": inputs["state"],
            "actions": self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs),
        }
        model_time = time.monotonic() - start_time
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    def _build_fkc_static_runtime(
        self, fkc_config: FKCConfig, norm_stats: dict[str, _transforms.NormStats]
    ) -> dict[str, jnp.ndarray]:
        """Pre-compute the FK + normalization tensors that don't change across infer calls."""
        action_stats = norm_stats["actions"]
        state_stats = norm_stats["state"]
        action_mean_7 = jnp.asarray(action_stats.mean[:7], dtype=jnp.float32)
        action_std_7 = jnp.asarray(action_stats.std[:7], dtype=jnp.float32)
        state_mean_7 = jnp.asarray(state_stats.mean[:7], dtype=jnp.float32)
        state_std_7 = jnp.asarray(state_stats.std[:7], dtype=jnp.float32)
        base_world_T, ee_offset_T = _fkc_fk.build_fk_transforms(fkc_config.fk)
        return {
            "action_mean_7": action_mean_7,
            "action_std_7": action_std_7,
            "state_mean_7": state_mean_7,
            "state_std_7": state_std_7,
            "base_world_T": base_world_T,
            "ee_offset_T": ee_offset_T,
            "target_xyz": jnp.asarray(fkc_config.cost.target_xyz, dtype=jnp.float32),
            "box_min_xyz": jnp.asarray(fkc_config.cost.box_min_xyz, dtype=jnp.float32),
            "box_max_xyz": jnp.asarray(fkc_config.cost.box_max_xyz, dtype=jnp.float32),
            "softplus_beta": jnp.asarray(fkc_config.cost.softplus_beta, dtype=jnp.float32),
        }

    def _materialise_fkc_runtime(self, observation: _model.Observation) -> _fkc_cc.FKRuntime:
        """Build the per-infer ``FKRuntime`` (un-normalizes the current joint state).

        ``current_joint_vel`` defaults to zero — the DROID observation schema
        doesn't carry joint velocity, so the actuator rollout starts from rest
        each chunk. The PD law settles within ~5 substeps so this initial-
        velocity error washes out quickly.
        """
        static = self._fkc_static_runtime_parts
        norm_state_7 = observation.state[..., :7]
        abs_joint_pos = norm_state_7 * (static["state_std_7"] + 1e-6) + static["state_mean_7"]
        zero_vel = jnp.zeros_like(abs_joint_pos)
        return _fkc_cc.FKRuntime(
            action_mean_7=static["action_mean_7"],
            action_std_7=static["action_std_7"],
            current_joint_pos=abs_joint_pos,
            current_joint_vel=zero_vel,
            base_world_T=static["base_world_T"],
            ee_offset_T=static["ee_offset_T"],
            target_xyz=static["target_xyz"],
            box_min_xyz=static["box_min_xyz"],
            box_max_xyz=static["box_max_xyz"],
            softplus_beta=static["softplus_beta"],
        )


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
