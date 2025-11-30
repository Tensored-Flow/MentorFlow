"""Evaluate a trained PPO student across all tasks and difficulties."""

import os
import sys
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO

# Ensure repository root is on sys.path for direct script execution.
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from training.evaluate import evaluate_model
from student.student_env import StudentEnv


def evaluate_all_tasks(
    model_path: str = "models/ppo_teacher_curriculum.zip",
    episodes: int = 100,
) -> np.ndarray:
    """Evaluate the saved PPO model on all task families/difficulties."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")

    accuracies = np.zeros((5, 3), dtype=float)
    for task_id in range(5):
        for diff in range(3):
            env = StudentEnv(family_id=task_id, difficulty_id=diff)
            model = PPO.load(model_path, env=env)
            acc = evaluate_model(
                model,
                episodes=episodes,
                family_id=task_id,
                difficulty_id=diff,
                env=env,
            )
            accuracies[task_id, diff] = acc
    return accuracies


def print_table(acc_matrix: np.ndarray) -> None:
    """Pretty-print a 5x3 accuracy table."""
    header = "Task/Diff | 0      | 1      | 2      "
    print(header)
    print("-" * len(header))
    for task_id in range(acc_matrix.shape[0]):
        row_vals: List[str] = [f"{acc_matrix[task_id, d]*100:6.2f}%" for d in range(3)]
        print(f"{task_id:9d} | " + " | ".join(row_vals))


def plot_heatmap(acc_matrix: np.ndarray, save_path: str = "demo_accuracy_overview.png") -> None:
    """Save a heatmap of accuracies."""
    plt.figure(figsize=(6, 5))
    plt.imshow(acc_matrix, vmin=0, vmax=1, cmap="viridis")
    plt.colorbar(label="accuracy")
    plt.xticks(range(3), [0, 1, 2])
    plt.yticks(range(5), [f"task {i}" for i in range(5)])
    plt.xlabel("difficulty")
    plt.ylabel("task")
    plt.title("Student accuracy across tasks/difficulties")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved accuracy heatmap to {save_path}")


def main() -> None:
    acc_matrix = evaluate_all_tasks()
    print("Accuracy table (percentage):")
    print_table(acc_matrix)
    plot_heatmap(acc_matrix)


if __name__ == "__main__":
    main()
