"""Teacher-guided PPO training across all task families and difficulties (multi-step, leak-safe)."""

import math
import os
import sys
from typing import Dict, List, Tuple

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
from teacher.teacher_ucb import TeacherUCB
from training.evaluate import evaluate_model

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


class MinibatchProgressCallback(BaseCallback):
    def __init__(self, total_timesteps: int, verbose: int = 0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.pbar: tqdm | None = None
        self.total_minibatches: int | None = None
        self._original_get = None
        self._patched = False

    def _on_training_start(self) -> None:
        model = self.model
        n_steps = model.n_steps
        n_envs = model.n_envs
        batch_size = model.batch_size
        n_epochs = model.n_epochs

        num_updates = max(1, math.ceil(self.total_timesteps / (n_steps * n_envs)))
        minibatches_per_update = math.ceil((n_steps * n_envs) / batch_size)
        self.total_minibatches = num_updates * n_epochs * minibatches_per_update

        self.pbar = tqdm(
            total=self.total_minibatches,
            desc="PPO updates",
            unit="mbatch",
            leave=False,
        )

    def _on_rollout_start(self) -> None:
        if self._patched or self.pbar is None:
            return
        buffer = self.model.rollout_buffer
        self._original_get = buffer.get

        def wrapped_get(*args, **kwargs):
            for batch in self._original_get(*args, **kwargs):
                if self.pbar is not None:
                    self.pbar.update(1)
                yield batch

        buffer.get = wrapped_get
        self._patched = True

    def _on_step(self) -> bool:
        return True

    def _on_training_end(self) -> None:
        if self._patched and self._original_get is not None:
            self.model.rollout_buffer.get = self._original_get
        if self.pbar is not None:
            if self.pbar.n < self.pbar.total:
                self.pbar.update(self.pbar.total - self.pbar.n)
            self.pbar.close()
        self.pbar = None
        self._original_get = None
        self._patched = False


def _arm_to_indices(arm_idx: int) -> Tuple[int, int]:
    task_id = arm_idx // 3
    difficulty = arm_idx % 3
    return task_id, difficulty


def train_with_teacher(
    total_rounds: int = 12,
    train_steps_per_round: int = 800,
    eval_episodes: int = 200,
    eval_task: Tuple[int, int] = (0, 0),
    save_path: str = "models/ppo_teacher_curriculum.zip",
) -> None:
    torch.set_num_threads(1)
    os.makedirs("models", exist_ok=True)

    num_tasks = 5
    num_difficulties = 3
    num_arms = num_tasks * num_difficulties
    arms = list(range(num_arms))
    teacher = TeacherUCB(num_arms=num_arms)

    model: PPO | None = None
    last_acc_per_arm: List[float] = [0.0 for _ in arms]
    eval_timesteps: List[int] = []
    eval_accuracies: List[float] = []
    cumulative_steps = 0
    round_history: List[Dict[str, float]] = []
    curriculum_counts = np.zeros((num_tasks, num_difficulties), dtype=np.int32)

    with tqdm(total=total_rounds, desc="Teacher rounds", unit="round") as round_pbar:
        for round_idx in range(total_rounds):
            arm_idx = teacher.select_arm()
            task_id, difficulty = _arm_to_indices(arm_idx)
            curriculum_counts[task_id, difficulty] += 1

            env = StudentEnv(family_id=task_id, difficulty_id=difficulty)
            if model is None:
                cfg = PPO_CFG.copy()
                model = PPO("MlpPolicy", env, verbose=0, **cfg)
            else:
                model.set_env(env)

            progress_cb = MinibatchProgressCallback(total_timesteps=train_steps_per_round)
            model.learn(
                total_timesteps=train_steps_per_round,
                reset_num_timesteps=False,
                callback=progress_cb,
            )
            cumulative_steps += train_steps_per_round

            prev_acc = last_acc_per_arm[arm_idx]
            arm_acc = evaluate_model(
                model, episodes=eval_episodes, family_id=task_id, difficulty_id=difficulty
            )
            reward = arm_acc - prev_acc
            teacher.update(arm_idx, reward)
            last_acc_per_arm[arm_idx] = arm_acc

            eval_task_id, eval_diff = eval_task
            eval_acc = evaluate_model(
                model,
                episodes=eval_episodes,
                family_id=eval_task_id,
                difficulty_id=eval_diff,
            )
            eval_timesteps.append(cumulative_steps)
            eval_accuracies.append(eval_acc)

            round_history.append(
                {
                    "round": round_idx + 1,
                    "arm": arm_idx,
                    "task_id": task_id,
                    "difficulty": difficulty,
                    "arm_acc": arm_acc,
                    "reward": reward,
                    "eval_acc": eval_acc,
                    "timesteps": cumulative_steps,
                }
            )

            round_pbar.update(1)
            round_pbar.set_postfix(
                arm=f"{task_id}/{difficulty}", arm_acc=f"{arm_acc:.2f}", eval=f"{eval_acc:.2f}"
            )

    if model is not None:
        model.save(save_path)
        print(f"Saved trained model to {save_path}")

    plt.figure(figsize=(8, 5))
    plt.plot(eval_timesteps, eval_accuracies, marker="o")
    plt.xlabel("timesteps")
    plt.ylabel("accuracy over eval task")
    plt.title("Teacher-guided PPO: evaluation accuracy")
    plt.grid(True)
    plt.tight_layout()
    plot_path = "models/teacher_overall_accuracy.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved evaluation plot to {plot_path}")

    plt.figure(figsize=(8, 4))
    rounds = [r["round"] for r in round_history]
    arm_accs = [r["arm_acc"] for r in round_history]
    rewards = [r["reward"] for r in round_history]
    plt.plot(rounds, arm_accs, marker="o", label="arm accuracy")
    plt.plot(rounds, rewards, marker="x", label="teacher reward (Δ acc)")
    plt.xlabel("round")
    plt.ylabel("value")
    plt.title("Per-round arm accuracy and teacher reward")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plot_path_rounds = "models/teacher_round_accuracy.png"
    plt.savefig(plot_path_rounds)
    plt.close()
    print(f"Saved round-by-round plot to {plot_path_rounds}")

    plt.figure(figsize=(6, 6))
    plt.imshow(curriculum_counts, cmap="Blues")
    plt.colorbar(label="times chosen")
    plt.xticks(range(num_difficulties), [0, 1, 2])
    plt.yticks(range(num_tasks), [f"task {i}" for i in range(num_tasks)])
    plt.xlabel("difficulty")
    plt.ylabel("task")
    plt.title("Curriculum heatmap (arm selections)")
    plt.tight_layout()
    heatmap_path = "models/teacher_curriculum_heatmap.png"
    plt.savefig(heatmap_path)
    plt.close()
    print(f"Saved curriculum heatmap to {heatmap_path}")

    summary_path = "models/teacher_training_summary.txt"
    with open(summary_path, "w") as f:
        f.write("Teacher training summary\n")
        f.write(f"total_rounds: {total_rounds}\n")
        f.write(f"train_steps_per_round: {train_steps_per_round}\n")
        if eval_accuracies:
            f.write(f"final_eval_accuracy: {eval_accuracies[-1]:.4f}\n")
            f.write(f"max_eval_accuracy: {max(eval_accuracies):.4f}\n")
        f.write(f"heatmap: {heatmap_path}\n")
        f.write(f"overall_accuracy_plot: {plot_path}\n")
        f.write(f"round_accuracy_plot: {plot_path_rounds}\n")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    train_with_teacher()
