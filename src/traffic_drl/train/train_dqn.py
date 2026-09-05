"""DQN training entry points for the Phase 1 pilot."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .checkpointing import load_checkpoint, save_checkpoint


def build_dqn_model(env: Any, config: Mapping[str, Any], *, seed: int | None = None, tensorboard_log: str | Path | None = None) -> Any:
    """Construct an SB3 DQN with a Gymnasium-compatible environment.

    Args:
        env: Single- or vectorized Gymnasium environment.
        config: DQN hyperparameters and policy settings.
        seed: Model and replay-buffer random seed.
        tensorboard_log: Optional training-log directory.

    TODO (SV2): restrict official training to TR scenarios.

    Returns:
        Any: An initialized SB3 DQN model.
    """
    raise NotImplementedError


def train_dqn(model: Any, *, total_timesteps: int, callback: Any | None = None, reset_num_timesteps: bool = True) -> Any:
    """Train a DQN model and return it.

    Args:
        model: Constructed SB3 DQN instance.
        total_timesteps: Number of environment transitions to collect.
        callback: Optional checkpoint or evaluation callback.
        reset_num_timesteps: Reset the counter for a fresh run; use false when resuming.

    Returns:
        Any: The trained or partially trained DQN model.
    """
    raise NotImplementedError


def save_dqn_checkpoint(model: Any, path: str | Path, *, replay_buffer_path: str | Path | None = None, vec_normalize_env: Any | None = None, resume_info: Mapping[str, Any] | None = None) -> None:
    """Save model, replay buffer, normalization stats, and resume metadata.

    Args:
        model: Trained SB3 DQN instance.
        path: Base ``.zip`` file containing serialized SB3 DQN state.
        replay_buffer_path: Optional SB3 replay-memory ``.pkl`` file.
        vec_normalize_env: VecNormalize state source, saved as sibling ``.pkl``.
        resume_info: Metadata serialized as sibling UTF-8 JSON.

    Returns:
        None: The model and requested companion files are written to disk.
    """
    save_checkpoint(model, path, vec_normalize_env=vec_normalize_env, save_replay_buffer=replay_buffer_path is not None, replay_buffer_path=replay_buffer_path, resume_info=resume_info)


def load_dqn_checkpoint(path: str | Path, *, env: Any | None = None, replay_buffer_path: str | Path | None = None, vec_normalize_stats: str | Path | None = None) -> Any:
    """Load a DQN checkpoint and optional training state.

    Args:
        path: Existing SB3 DQN ``.zip`` parsed by ``DQN.load``.
        env: Fresh compatible environment, required with normalization stats.
        replay_buffer_path: Optional replay-memory ``.pkl`` parsed by
            ``load_replay_buffer``.
        vec_normalize_stats: Optional VecNormalize ``.pkl`` parsed before the model.

    Returns:
        Any: A loaded SB3 DQN ready for inference or continued learning.
    """
    from stable_baselines3 import DQN

    return load_checkpoint(DQN, path, env=env, vec_normalize_path=vec_normalize_stats, replay_buffer_path=replay_buffer_path)


def run_dqn_pilot(config: Mapping[str, Any], *, checkpoint_dir: str | Path, total_timesteps: int = 10_000, seed: int = 5) -> Any:
    """Run the DEV-00 pilot without treating it as a final result.

    Args:
        config: Pilot environment and DQN settings.
        checkpoint_dir: Directory containing model ``.zip``, replay ``.pkl``,
            VecNormalize ``.pkl``, and resume ``.json`` artifacts.
        total_timesteps: Pilot training budget.
        seed: Development seed.

    TODO (SV2): verify reset/step, finite rewards, reload, and legal actions.

    Returns:
        Any: The pilot DQN model and its checkpoint state.
    """
    raise NotImplementedError
