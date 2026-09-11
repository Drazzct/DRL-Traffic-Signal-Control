"""Stable-Baselines3 DQN training entry points for the Phase 1 pilot.

The public API mirrors the PPO training module
(:mod:`traffic_drl.train.train_ppo`) with the addition of replay-buffer
checkpointing.

Typical usage
-------------
::

    from traffic_drl.config import load_train_config
    from traffic_drl.train.train_dqn import build_dqn_model, train_dqn
    from traffic_drl.train.callbacks import RobustCheckpointCallback

    cfg = load_train_config("configs/train/dqn_phase1.yaml")
    model = build_dqn_model(env, cfg, seed=cfg.experiment.seed)
    callback = RobustCheckpointCallback(save_freq=cfg.training_control.save_freq,
                                        save_dir="outputs/checkpoints")
    model = train_dqn(model, total_timesteps=cfg.training_control.total_timesteps,
                      callback=callback)
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from .checkpointing import load_checkpoint, save_checkpoint
from ..config import TrainConfig, load_train_config

if TYPE_CHECKING:
    from stable_baselines3 import DQN
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


def build_dqn_model(
    env: "DummyVecEnv | VecNormalize",
    config: TrainConfig,
    *,
    seed: int | None = None,
    tensorboard_log: str | Path | None = None,
) -> "DQN":
    """Construct an SB3 DQN with a vectorised Gymnasium-compatible environment.

    Reads ``config.model_hyperparameters`` to set the policy, learning rate,
    batch size, buffer size, and all other SB3 DQN constructor kwargs.

    Args:
        env: A vectorised (``DummyVecEnv``) and optionally normalised
            (``VecNormalize``) environment from
            :func:`~traffic_drl.environment.make_env.make_vectorized_environment`.
        config: Typed training configuration from
            :func:`~traffic_drl.config.load_train_config`.
        seed: Model and replay-buffer random seed.  Overrides
            ``config.experiment.seed`` if provided.
        tensorboard_log: Optional TensorBoard log directory; passed directly
            to the SB3 ``DQN`` constructor.

    TODO (SV2): map every ``config.model_hyperparameters`` key to the SB3 DQN
    constructor; restrict official training to ``TR`` scenarios only.

    Returns:
        DQN: An initialised SB3 DQN model ready for :func:`train_dqn`.
    """
    raise NotImplementedError


def train_dqn(
    model: "DQN",
    *,
    total_timesteps: int,
    callback: "BaseCallback | None" = None,
    reset_num_timesteps: bool = True,
) -> "DQN":
    """Train a DQN model and return it.

    Args:
        model: Constructed SB3 DQN instance from :func:`build_dqn_model`.
        total_timesteps: Number of environment transitions to collect.
        callback: Optional SB3 callback, e.g.
            :class:`~traffic_drl.train.callbacks.RobustCheckpointCallback`.
        reset_num_timesteps: ``True`` for a fresh run; ``False`` when resuming
            from a checkpoint.

    TODO (SV2): call ``model.learn(total_timesteps=total_timesteps, callback=callback,
    reset_num_timesteps=reset_num_timesteps)`` and return the trained model.

    Returns:
        DQN: The trained (or partially trained) DQN model.
    """
    raise NotImplementedError


def save_dqn_checkpoint(
    model: "DQN",
    path: str | Path,
    *,
    replay_buffer_path: str | Path | None = None,
    vec_normalize_env: "VecNormalize | None" = None,
    resume_info: dict[str, str | int | float | bool] | None = None,
) -> None:
    """Save model, replay buffer, normalisation stats, and resume metadata.

    Delegates to :func:`~traffic_drl.train.checkpointing.save_checkpoint`.

    Args:
        model: Trained SB3 DQN instance.
        path: Base ``.zip`` checkpoint file.
        replay_buffer_path: Optional ``.pkl`` replay-buffer destination.
        vec_normalize_env: ``VecNormalize`` wrapper whose statistics are saved
            as a sibling ``.pkl`` file.
        resume_info: Optional metadata dict written as sibling JSON.

    Returns:
        None.
    """
    save_checkpoint(
        model,
        path,
        vec_normalize_env=vec_normalize_env,
        save_replay_buffer=replay_buffer_path is not None,
        replay_buffer_path=replay_buffer_path,
        resume_info=resume_info,
    )


def load_dqn_checkpoint(
    path: str | Path,
    *,
    env: "DummyVecEnv | VecNormalize | None" = None,
    replay_buffer_path: str | Path | None = None,
    vec_normalize_stats: str | Path | None = None,
) -> "DQN":
    """Load a DQN checkpoint and optional training state.

    Args:
        path: Existing SB3 DQN ``.zip`` checkpoint.
        env: Fresh compatible environment; required when loading normalisation
            statistics.
        replay_buffer_path: Optional ``.pkl`` replay-buffer file.
        vec_normalize_stats: Optional ``VecNormalize`` ``.pkl`` file to restore
            before the model.

    Returns:
        DQN: A loaded SB3 DQN ready for inference or continued training.
    """
    from stable_baselines3 import DQN as _DQN

    return load_checkpoint(
        _DQN,
        path,
        env=env,
        vec_normalize_path=vec_normalize_stats,
        replay_buffer_path=replay_buffer_path,
    )


def run_dqn_pilot(
    config: TrainConfig | str | Path = "configs/train/dqn_phase1.yaml",
    *,
    checkpoint_dir: str | Path,
    total_timesteps: int = 10_000,
    seed: int = 5,
) -> "DQN":
    """Run the DEV-00 pilot training without treating it as a final result.

    Loads config, builds the environment, constructs the DQN, trains, and
    saves a checkpoint bundle.  The result is a development sanity check only.

    Args:
        config: Typed training configuration or path to its YAML file.
        checkpoint_dir: Directory for model, replay-buffer, normalisation, and
            resume metadata artifacts.
        total_timesteps: Pilot training budget (intentionally small).
        seed: Development seed; overrides ``config.experiment.seed``.

    TODO (SV2): verify reset/step contract, finite rewards, checkpoint reload,
    and legal action sampling before declaring the pilot complete.

    Returns:
        DQN: The pilot DQN model after training.
    """
    if isinstance(config, (str, Path)):
        config = load_train_config(config)

    raise NotImplementedError
