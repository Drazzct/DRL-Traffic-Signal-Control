"""Stable-Baselines3 callbacks for checkpoints and traffic metrics."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from stable_baselines3.common.callbacks import BaseCallback

from .checkpointing import save_checkpoint


class RobustCheckpointCallback(BaseCallback):
    """Save model, replay buffer, VecNormalize and resume metadata."""

    def __init__(self, save_freq: int, save_dir: str | Path, *, save_replay_buffer: bool = True, save_vec_normalize: bool = True, verbose: int = 0) -> None:
        super().__init__(verbose=verbose)
        self.save_freq = save_freq
        self.save_dir = Path(save_dir)
        self.save_replay_buffer = save_replay_buffer
        self.save_vec_normalize = save_vec_normalize

    def _on_step(self) -> bool:
        """Save artifacts at the configured SB3 callback interval.

        Returns:
            bool: ``True`` to continue training; ``False`` to stop training.
        """
        if self.n_calls % self.save_freq != 0:
            return True

        save_checkpoint(
            self.model,
            self.save_dir / f"rl_model_{self.num_timesteps}",
            vec_normalize_env=self.training_env if self.save_vec_normalize and hasattr(self.training_env, "save") else None,
            save_replay_buffer=self.save_replay_buffer,
            resume_info={"num_timesteps": self.num_timesteps, "model_class": self.model.__class__.__name__},
        )
        return True

    def _on_training_end(self) -> None:
        """Persist final resume metadata.

        Returns:
            None: The final checkpoint bundle is written to ``save_dir``.
        """
        save_checkpoint(
            self.model,
            self.save_dir / "latest",
            vec_normalize_env=self.training_env if self.save_vec_normalize and hasattr(self.training_env, "save") else None,
            save_replay_buffer=self.save_replay_buffer,
            resume_info={"num_timesteps": self.num_timesteps, "model_class": self.model.__class__.__name__},
        )


class TrafficMetricsCallback(BaseCallback):
    """Copy episode traffic metrics from ``info`` into SB3's logger."""

    def _on_step(self) -> bool:
        """Log waiting, queue, timeLoss, throughput and control metrics.

        Returns:
            bool: ``True`` to continue training; ``False`` to stop training.
        """
        for info in self.locals.get("infos", []):
            for metric in ("average_waiting_time", "average_queue_length", "time_loss", "throughput", "phase_switch_rate", "min_green_violations"):
                if metric in info:
                    self.logger.record(f"traffic/{metric}", info[metric])
        return True


class Phase1PilotCallback(RobustCheckpointCallback):
    """Checkpoint callback with pilot-specific metadata."""

    def _on_step(self) -> bool:
        """Save a pilot checkpoint while preserving the SB3 callback contract.

        Returns:
            bool: Continuation decision returned by the parent callback.
        """
        return super()._on_step()
