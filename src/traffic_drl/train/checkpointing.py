"""Shared Stable-Baselines3 checkpoint persistence utilities."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from traffic_drl.contracts import ResumeInfo


class CheckpointBundle:
    """Paths for one model checkpoint and its optional resume artifacts."""

    def __init__(self, model_path: str | Path) -> None:
        """Create derived artifact paths for ``model_path``.

        Args:
            model_path: Base SB3 ``.zip`` checkpoint containing serialized policy,
                optimizer, spaces, and hyperparameters; parse it with the
                algorithm's ``load`` method.
        """
        self.model_path = Path(model_path)
        self.replay_buffer_path = self.model_path.with_name(self.model_path.stem + "_replay_buffer.pkl")
        self.vec_normalize_path = self.model_path.with_name(self.model_path.stem + "_vecnormalize.pkl")
        self.resume_info_path = self.model_path.with_name(self.model_path.stem + "_resume_info.json")

    def ensure_parent(self) -> None:
        """Create the checkpoint directory if necessary.

        Returns:
            None: The directory exists after this method returns.
        """
        self.model_path.parent.mkdir(parents=True, exist_ok=True)


def save_checkpoint(
    model: Any,
    model_path: str | Path,
    *,
    vec_normalize_env: Any | None = None,
    save_replay_buffer: bool = False,
    replay_buffer_path: str | Path | None = None,
    resume_info: ResumeInfo | Mapping[str, Any] | None = None,
) -> CheckpointBundle:
    """Save an SB3 model and optional artifacts as one named bundle.

    Args:
        model: SB3 algorithm instance exposing ``save``.
        model_path: Destination SB3 ``.zip`` file.
        vec_normalize_env: VecNormalize wrapper whose running mean, variance,
            clipping, and reward-normalization state are saved to ``.pkl``.
        save_replay_buffer: Save replay memory when supported by the model.
        replay_buffer_path: Optional ``.pkl`` replay-memory destination.
        resume_info: Metadata written as UTF-8 JSON for resuming training.

    TODO (SV2): verify the bundle contains model, normalizer, config, and resume info.

    Returns:
        CheckpointBundle: Paths for the model and all saved companion artifacts.
    """
    bundle = CheckpointBundle(model_path)
    bundle.ensure_parent()
    model.save(str(bundle.model_path))

    if save_replay_buffer and hasattr(model, "save_replay_buffer"):
        replay_path = Path(replay_buffer_path) if replay_buffer_path is not None else bundle.replay_buffer_path
        replay_path.parent.mkdir(parents=True, exist_ok=True)
        model.save_replay_buffer(str(replay_path))

    if vec_normalize_env is not None:
        vec_normalize_env.save(str(bundle.vec_normalize_path))

    if resume_info is not None:
        payload = resume_info.__dict__ if isinstance(resume_info, ResumeInfo) else dict(resume_info)
        bundle.resume_info_path.write_text(
            json.dumps(payload, indent=2, default=str),
            encoding="utf-8",
        )

    return bundle


def load_checkpoint(
    model_class: Any,
    model_path: str | Path,
    *,
    env: Any | None = None,
    vec_normalize_path: str | Path | None = None,
    replay_buffer_path: str | Path | None = None,
    training: bool = True,
    norm_reward: bool = False,
) -> Any:
    """Load an SB3 model after restoring optional VecNormalize statistics.

    Args:
        model_class: SB3 class such as ``DQN`` or ``PPO``.
        model_path: Existing SB3 ``.zip`` checkpoint parsed by ``model_class.load``.
        env: Fresh environment compatible with the saved model.
        vec_normalize_path: Saved VecNormalize ``.pkl`` parsed with
            ``VecNormalize.load`` before the model.
        replay_buffer_path: Optional DQN replay-buffer ``.pkl`` parsed with
            ``model.load_replay_buffer``.
        training: Keep normalization statistics updating when true.
        norm_reward: Restore reward normalization mode.

    TODO (SV2): run one short reset/step after loading before resuming long training.

    Returns:
        Any: The loaded SB3 model, optionally attached to restored VecNormalize.
    """
    load_env = env
    if vec_normalize_path is not None:
        if env is None:
            raise ValueError("env is required when loading VecNormalize statistics")
        from stable_baselines3.common.vec_env import VecNormalize

        load_env = VecNormalize.load(str(vec_normalize_path), env)
        load_env.training = training
        load_env.norm_reward = norm_reward

    model = model_class.load(str(model_path), env=load_env)
    if replay_buffer_path is not None:
        if not hasattr(model, "load_replay_buffer"):
            raise TypeError(f"{model_class.__name__} does not support replay-buffer loading")
        model.load_replay_buffer(str(replay_buffer_path))
    return model


def read_resume_info(model_path: str | Path) -> ResumeInfo:
    """Read resume metadata associated with a checkpoint bundle.

    Args:
        model_path: Base model ``.zip`` checkpoint whose sibling
            ``*_resume_info.json`` file is parsed with :mod:`json`.

    Returns:
        ResumeInfo: Typed reproducibility metadata from the checkpoint bundle.
    """
    bundle = CheckpointBundle(model_path)
    payload = json.loads(bundle.resume_info_path.read_text(encoding="utf-8"))
    return ResumeInfo(
        num_timesteps=int(payload["num_timesteps"]),
        model_class=str(payload["model_class"]),
        seed=payload.get("seed"),
        scenario_split=payload.get("scenario_split"),
        config_path=Path(payload["config_path"]) if payload.get("config_path") else None,
        git_commit=payload.get("git_commit"),
        extra=payload.get("extra", {}),
    )
