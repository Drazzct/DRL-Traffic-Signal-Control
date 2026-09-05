"""PPO training API reserved for the later Phase 2 pipeline."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .checkpointing import load_checkpoint, save_checkpoint


def build_ppo_model(env: Any, config: Mapping[str, Any], *, seed: int | None = None, tensorboard_log: str | Path | None = None) -> Any:
    """Construct an SB3 PPO model.

    Args:
        env: Gymnasium environment wrapped for SB3.
        config: PPO hyperparameters and policy settings.
        seed: Model random seed.
        tensorboard_log: Optional training-log directory.

    TODO (SV2): keep PPO secondary to the minimum DQN deliverable.

    Returns:
        Any: An initialized SB3 PPO model.
    """
    raise NotImplementedError


def train_ppo(model: Any, *, total_timesteps: int, callback: Any | None = None, reset_num_timesteps: bool = True) -> Any:
    """Train PPO and return the model.

    Args:
        model: Constructed SB3 PPO instance.
        total_timesteps: Number of environment transitions.
        callback: Optional checkpoint or evaluation callback.
        reset_num_timesteps: Reset or continue SB3's timestep counter.

    Returns:
        Any: The trained or partially trained PPO model.
    """
    raise NotImplementedError


def save_ppo_checkpoint(model: Any, path: str | Path, *, vec_normalize_env: Any | None = None, resume_info: Mapping[str, Any] | None = None) -> None:
    """Save PPO and all artifacts required for resuming.

    Args:
        model: Trained SB3 PPO instance.
        path: Base ``.zip`` file containing serialized SB3 PPO state.
        vec_normalize_env: VecNormalize state source, saved as sibling ``.pkl``.
        resume_info: Metadata serialized as sibling UTF-8 JSON.

    Returns:
        None: The model and requested companion files are written to disk.
    """
    save_checkpoint(model, path, vec_normalize_env=vec_normalize_env, resume_info=resume_info)


def load_ppo_checkpoint(path: str | Path, *, env: Any | None = None, vec_normalize_stats: str | Path | None = None) -> Any:
    """Load a PPO checkpoint with optional normalization statistics.

    Args:
        path: Existing PPO ``.zip`` parsed by ``PPO.load``.
        env: Fresh compatible environment, required with normalization stats.
        vec_normalize_stats: Saved VecNormalize ``.pkl`` parsed before the model.

    Returns:
        Any: A loaded SB3 PPO ready for inference or continued learning.
    """
    from stable_baselines3 import PPO

    return load_checkpoint(PPO, path, env=env, vec_normalize_path=vec_normalize_stats)
