"""Shared Stable-Baselines3 checkpoint persistence utilities.

One checkpoint bundle = one SB3 model ``.zip`` + optional sibling files:
- ``<stem>_replay_buffer.pkl`` — DQN replay memory.
- ``<stem>_vecnormalize.pkl`` — ``VecNormalize`` running statistics.
- ``<stem>_resume_info.json`` — metadata for reproducible resumption.

Typical usage
-------------
::

    from traffic_drl.train.checkpointing import save_checkpoint, load_checkpoint
    from stable_baselines3 import DQN

    bundle = save_checkpoint(model, "outputs/checkpoints/step_10000",
                             vec_normalize_env=vec_env,
                             save_replay_buffer=True,
                             resume_info={"num_timesteps": 10000, "model_class": "DQN"})

    loaded = load_checkpoint(DQN, bundle.model_path, env=fresh_env,
                             vec_normalize_path=bundle.vec_normalize_path)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

from traffic_drl.contracts import ResumeInfo

if TYPE_CHECKING:
    from stable_baselines3 import DQN, PPO
    from stable_baselines3.common.vec_env import VecNormalize

# Generic SB3 model type — either DQN or PPO.
# Used to preserve the concrete type through load_checkpoint's return.
_SB3ModelT = TypeVar("_SB3ModelT")


class CheckpointBundle:
    """Paths for one SB3 model checkpoint and its optional companion artifacts.

    Constructed by :func:`save_checkpoint`; passed to :func:`load_checkpoint`.

    Attributes:
        model_path: Base ``.zip`` file containing the serialised SB3 policy,
            optimizer state, and hyperparameters.
        replay_buffer_path: Sibling ``.pkl`` for DQN replay memory.
        vec_normalize_path: Sibling ``.pkl`` for ``VecNormalize`` statistics.
        resume_info_path: Sibling ``.json`` for reproducibility metadata.
    """

    def __init__(self, model_path: str | Path) -> None:
        """Derive all sibling paths from *model_path*.

        Args:
            model_path: Base checkpoint path (with or without ``.zip`` suffix).
        """
        self.model_path = Path(model_path)
        stem = self.model_path.stem
        parent = self.model_path.parent
        self.replay_buffer_path: Path = parent / f"{stem}_replay_buffer.pkl"
        self.vec_normalize_path: Path = parent / f"{stem}_vecnormalize.pkl"
        self.resume_info_path: Path = parent / f"{stem}_resume_info.json"

    def ensure_parent(self) -> None:
        """Create the checkpoint directory if it does not already exist.

        Returns:
            None.
        """
        self.model_path.parent.mkdir(parents=True, exist_ok=True)


def save_checkpoint(
    model: "DQN | PPO",
    model_path: str | Path,
    *,
    vec_normalize_env: "VecNormalize | None" = None,
    save_replay_buffer: bool = False,
    replay_buffer_path: str | Path | None = None,
    resume_info: ResumeInfo | dict[str, str | int | float | bool] | None = None,
) -> CheckpointBundle:
    """Save an SB3 model and optional companion artifacts as one named bundle.

    Args:
        model: An SB3 algorithm instance (``DQN`` or ``PPO``) exposing
            ``.save()`` and optionally ``.save_replay_buffer()``.
        model_path: Destination ``.zip`` base path.
        vec_normalize_env: ``VecNormalize`` wrapper whose running mean,
            variance, clipping, and reward-normalisation state are saved to
            the sibling ``.pkl``.
        save_replay_buffer: Save replay memory when the model supports it
            (DQN only).
        replay_buffer_path: Optional explicit ``.pkl`` destination for the
            replay buffer; defaults to the sibling path from the bundle.
        resume_info: Reproducibility metadata written as UTF-8 JSON.

    TODO (SV2): verify the bundle contains model, normaliser, config, and
    resume info before declaring the checkpointing contract complete.

    Returns:
        CheckpointBundle: Paths for the model and all saved companion artifacts.
    """
    bundle = CheckpointBundle(model_path)
    bundle.ensure_parent()
    model.save(str(bundle.model_path))

    if save_replay_buffer and hasattr(model, "save_replay_buffer"):
        rb_path = Path(replay_buffer_path) if replay_buffer_path is not None else bundle.replay_buffer_path
        rb_path.parent.mkdir(parents=True, exist_ok=True)
        model.save_replay_buffer(str(rb_path))

    if vec_normalize_env is not None:
        vec_normalize_env.save(str(bundle.vec_normalize_path))

    if resume_info is not None:
        payload = (
            resume_info.__dict__
            if isinstance(resume_info, ResumeInfo)
            else dict(resume_info)
        )
        bundle.resume_info_path.write_text(
            json.dumps(payload, indent=2, default=str),
            encoding="utf-8",
        )

    return bundle


def load_checkpoint(
    model_class: "type[_SB3ModelT]",
    model_path: str | Path,
    *,
    env: "object | None" = None,
    vec_normalize_path: str | Path | None = None,
    replay_buffer_path: str | Path | None = None,
    training: bool = True,
    norm_reward: bool = False,
) -> "_SB3ModelT":
    """Load an SB3 model after restoring optional ``VecNormalize`` statistics.

    Args:
        model_class: The SB3 class to load with, e.g. ``DQN`` or ``PPO``.
        model_path: Existing ``.zip`` checkpoint.
        env: Fresh environment compatible with the saved model.  Required when
            *vec_normalize_path* is provided.
        vec_normalize_path: Saved ``VecNormalize`` ``.pkl`` to restore before
            the model.
        replay_buffer_path: Optional DQN replay-buffer ``.pkl``.
        training: Keep normalisation statistics updating when ``True`` (for
            resumed training); ``False`` for evaluation.
        norm_reward: Restore reward normalisation mode.

    TODO (SV2): run one short reset/step after loading before resuming a long
    training run to verify the environment and model are compatible.

    Returns:
        An instance of *model_class* loaded from *model_path*.
    """
    load_env = env
    if vec_normalize_path is not None:
        if env is None:
            raise ValueError("env is required when loading VecNormalize statistics.")
        from stable_baselines3.common.vec_env import VecNormalize

        load_env = VecNormalize.load(str(vec_normalize_path), env)
        load_env.training = training
        load_env.norm_reward = norm_reward

    loaded_model = model_class.load(str(model_path), env=load_env)
    if replay_buffer_path is not None:
        if not hasattr(loaded_model, "load_replay_buffer"):
            raise TypeError(
                f"{model_class.__name__} does not support replay-buffer loading."
            )
        loaded_model.load_replay_buffer(str(replay_buffer_path))
    return loaded_model


def read_resume_info(model_path: str | Path) -> ResumeInfo:
    """Read resume metadata associated with a checkpoint bundle.

    Args:
        model_path: Base model ``.zip`` checkpoint whose sibling
            ``*_resume_info.json`` is read.

    Raises:
        FileNotFoundError: If the resume-info JSON does not exist.
        KeyError: If required fields are missing from the JSON.

    Returns:
        ResumeInfo: Typed reproducibility metadata from the checkpoint bundle.
    """
    bundle = CheckpointBundle(model_path)
    if not bundle.resume_info_path.exists():
        raise FileNotFoundError(
            f"Resume info not found: {bundle.resume_info_path}"
        )
    payload = json.loads(bundle.resume_info_path.read_text(encoding="utf-8"))
    return ResumeInfo(
        num_timesteps=int(payload["num_timesteps"]),
        model_class=str(payload["model_class"]),
        seed=payload.get("seed"),
        scenario_split=payload.get("scenario_split"),
        config_path=Path(payload["config_path"]) if payload.get("config_path") else None,
        git_commit=payload.get("git_commit"),
        extra={k: v for k, v in payload.get("extra", {}).items()},
    )
