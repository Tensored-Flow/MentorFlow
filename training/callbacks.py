"""
Improved progress callbacks for Stable-Baselines3 training.

Fixes the "frozen progress bar" issue by tracking progress based on
rollout events (_on_rollout_start, _on_rollout_end) and model.num_timesteps
rather than relying on _on_step() which is only called once per rollout.
"""

from typing import Optional
from stable_baselines3.common.callbacks import BaseCallback
from tqdm import tqdm


class RolloutProgressCallback(BaseCallback):
    """
    Progress callback that tracks true timesteps by monitoring rollouts.
    
    Stable-Baselines3 only calls _on_step() once per rollout completion,
    not per environment step. This callback fixes the frozen progress bar
    by updating based on:
    - _on_rollout_start(): Record starting timesteps
    - _on_rollout_end(): Update progress bar with delta
    - _on_training_end(): Finalize progress bar
    """
    
    def __init__(self, total_timesteps: int, desc: str = "Training", verbose: int = 0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.desc = desc
        self.pbar: Optional[tqdm] = None
        self._last_timesteps = 0
        
    def _on_training_start(self) -> None:
        """Initialize progress bar at training start."""
        self.pbar = tqdm(
            total=self.total_timesteps,
            desc=self.desc,
            unit="step",
            initial=0,
        )
        self._last_timesteps = self.model.num_timesteps
        
    def _on_rollout_start(self) -> None:
        """Record timesteps at rollout start."""
        # Track the current number of timesteps when rollout starts
        self._last_timesteps = self.model.num_timesteps
        
    def _on_rollout_end(self) -> None:
        """Update progress bar when rollout completes."""
        if self.pbar is None:
            return
            
        current_timesteps = self.model.num_timesteps
        delta = current_timesteps - self._last_timesteps
        
        if delta > 0:
            # Update progress by the delta, but don't exceed total
            current_progress = self.pbar.n
            remaining = self.total_timesteps - current_progress
            update_amount = min(delta, remaining)
            if update_amount > 0:
                self.pbar.update(update_amount)
            
        self._last_timesteps = current_timesteps
        
    def _on_step(self) -> bool:
        """
        Called once per rollout, not per step.
        
        We don't rely on this for progress updates since it's unreliable
        for large n_steps values. Instead, we use rollout callbacks.
        """
        return True
        
    def _on_training_end(self) -> None:
        """Finalize progress bar at training end."""
        if self.pbar is not None:
            # Ensure we've reached the total
            current_timesteps = self.model.num_timesteps
            if current_timesteps < self.total_timesteps:
                remaining = self.total_timesteps - self.pbar.n
                if remaining > 0:
                    self.pbar.update(remaining)
            elif self.pbar.n < self.total_timesteps:
                # Update to exactly total_timesteps if we're close
                self.pbar.update(self.total_timesteps - self.pbar.n)
            self.pbar.close()
            self.pbar = None


class SharedProgressCallback(BaseCallback):
    """
    Progress callback that shares an external tqdm progress bar.
    
    Useful when you want to control the progress bar from outside the callback,
    such as in train_with_eval_logging where we have nested loops.
    """
    
    def __init__(self, pbar: tqdm, verbose: int = 0):
        super().__init__(verbose)
        self.pbar = pbar
        self._last_timesteps = 0
        
    def _on_training_start(self) -> None:
        """Record initial timesteps."""
        self._last_timesteps = self.model.num_timesteps
        
    def _on_rollout_start(self) -> None:
        """Record timesteps at rollout start."""
        self._last_timesteps = self.model.num_timesteps
        
    def _on_rollout_end(self) -> None:
        """Update shared progress bar when rollout completes."""
        current_timesteps = self.model.num_timesteps
        delta = current_timesteps - self._last_timesteps
        
        if delta > 0:
            # Update progress by the delta, but don't exceed total
            current_progress = self.pbar.n
            remaining = self.pbar.total - current_progress
            update_amount = min(delta, remaining)
            if update_amount > 0:
                self.pbar.update(update_amount)
            
        self._last_timesteps = current_timesteps
        
    def _on_step(self) -> bool:
        """Called once per rollout, not per step."""
        return True
        
    def _on_training_end(self) -> None:
        """Finalize progress on shared bar."""
        # Don't close the shared bar, let the caller handle it
        current_timesteps = self.model.num_timesteps
        if current_timesteps < self.pbar.total:
            remaining = min(self.pbar.total - self.pbar.n, self.pbar.total)
            if remaining > 0:
                self.pbar.update(remaining)
        self._last_timesteps = 0

