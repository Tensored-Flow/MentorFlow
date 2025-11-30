"""Train PPO on a single task family/difficulty with multi-step env (no leakage)."""

import os
import sys
from typing import Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from tqdm import tqdm

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from student.student_env import StudentEnv
from training.evaluate import evaluate_model
from training.callbacks import RolloutProgressCallback


PPO_CFG = dict(
    learning_rate=3e-4,
    n_steps=256,
    batch_size=256,
    n_epochs=4,
    gamma=0.95,
    gae_lambda=0.92,
    clip_range=0.2,
    ent_coef=0.01,
    policy_kwargs={"net_arch": [128, 128, 64]},
)


class RewardTrackerCallback(BaseCallback):
    def __init__(self, smooth_window: int = 50, verbose: int = 0):
        super().__init__(verbose)
        self.smooth_window = smooth_window
        self.rewards: List[float] = []
        self.timesteps: List[int] = []

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards")
        if rewards is not None:
            self.timesteps.append(self.model.num_timesteps)
            self.rewards.append(float(np.mean(rewards)))
        return True

    def _smooth(self, values: Iterable[float]) -> List[float]:
        arr = np.array(list(values), dtype=float)
        if arr.size == 0:
            return []
        window = self.smooth_window
        if window <= 1 or arr.size < window:
            return arr.tolist()
        cumsum = np.cumsum(np.insert(arr, 0, 0))
        smoothed = (cumsum[window:] - cumsum[:-window]) / window
        pad = [smoothed[0]] * (window - 1)
        return pad + smoothed.tolist()

    def save_plot(self, path: str) -> None:
        if not self.timesteps:
            return
        smoothed = self._smooth(self.rewards)
        plt.figure(figsize=(6, 4))
        plt.plot(self.timesteps, self.rewards, label="reward per step", alpha=0.5)
        plt.plot(self.timesteps, smoothed, label=f"smoothed (w={self.smooth_window})", linewidth=2)
        plt.xlabel("timesteps")
        plt.ylabel("reward")
        plt.title("Training reward")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(path)
        plt.close()


def train_single_task(
    family_id: int = 0,
    difficulty_id: int = 0,
    total_timesteps: int = 50_000,
    eval_episodes: int = 500,
    save_dir: str = "models",
) -> PPO:
    torch.set_num_threads(1)
    os.makedirs(save_dir, exist_ok=True)

    env = StudentEnv(family_id=family_id, difficulty_id=difficulty_id)
    cfg = PPO_CFG.copy()
    
    # Ensure n_steps is reasonable - for very small runs, use smaller rollouts
    # PPO needs to collect at least n_steps before it can start learning
    if total_timesteps < 256:
        cfg["n_steps"] = max(32, total_timesteps // 4)  # Use 1/4 of total for rollout
        cfg["batch_size"] = max(32, cfg["n_steps"])  # Batch size shouldn't be larger than n_steps
    else:
        # Clamp rollout/batch sizes so very small runs don't hang collecting long rollouts.
        cfg["n_steps"] = min(cfg["n_steps"], total_timesteps)
        cfg["batch_size"] = min(cfg["batch_size"], total_timesteps)
    
    print(f"Training configuration:")
    print(f"  n_steps (rollout length): {cfg['n_steps']}")
    print(f"  batch_size: {cfg['batch_size']}")
    print(f"  total_timesteps: {total_timesteps}")
    
    model = PPO("MlpPolicy", env, verbose=0, **cfg)

    progress_cb = RolloutProgressCallback(
        total_timesteps=total_timesteps,
        desc=f"Training family={family_id} diff={difficulty_id}"
    )
    reward_cb = RewardTrackerCallback(smooth_window=200)
    print(f"\nTraining PPO on family={family_id}, difficulty={difficulty_id} for {total_timesteps} steps...")
    print("(Progress bar will update after each rollout completes)\n")
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=[progress_cb, reward_cb],
        progress_bar=False,
    )
    
    print("\n✓ Training completed!")

    model_path = os.path.join(save_dir, f"ppo_family{family_id}_diff{difficulty_id}.zip")
    model.save(model_path)
    print(f"Saved model to {model_path}")

    eval_eps = min(eval_episodes, max(50, total_timesteps // 10))  # Scale eval with training size
    print(f"\nEvaluating over {eval_eps} episodes...")
    acc = evaluate_model(model, episodes=eval_eps, family_id=family_id, difficulty_id=difficulty_id)
    reward_plot = os.path.join(save_dir, f"family{family_id}_diff{difficulty_id}_reward.png")
    reward_cb.save_plot(reward_plot)

    plt.figure(figsize=(5, 3))
    plt.plot([total_timesteps], [acc], marker="o")
    plt.xlabel("timesteps")
    plt.ylabel("accuracy")
    plt.title("Evaluation accuracy")
    plt.ylim(0, 1)
    plt.tight_layout()
    acc_plot = os.path.join(save_dir, f"family{family_id}_diff{difficulty_id}_accuracy.png")
    plt.savefig(acc_plot)
    plt.close()
    print(f"Evaluation accuracy over {eval_eps} episodes: {acc * 100:.2f}%")

    summary_path = os.path.join(save_dir, f"family{family_id}_diff{difficulty_id}_summary.txt")
    with open(summary_path, "w") as f:
        f.write("Single-task PPO training summary\n")
        f.write(f"family_id: {family_id}\n")
        f.write(f"difficulty_id: {difficulty_id}\n")
        f.write(f"total_timesteps: {total_timesteps}\n")
        f.write(f"final_accuracy: {acc:.4f}\n")
        f.write(f"reward_plot: {reward_plot}\n")
        f.write(f"acc_plot: {acc_plot}\n")
    print(f"Saved reward plot to {reward_plot}")
    print(f"Saved accuracy plot to {acc_plot}")
    print(f"Saved summary to {summary_path}")

    return model


if __name__ == "__main__":
    # Quick test with small timesteps - increase for real training
    train_single_task(family_id=0, difficulty_id=0, total_timesteps=1000, eval_episodes=100)
