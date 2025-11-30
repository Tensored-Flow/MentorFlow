"""Train a lightweight LM-based student with a frozen GPT-2 encoder and PPO."""

import os
import sys
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from stable_baselines3 import PPO
from transformers import AutoTokenizer

# Ensure repository root on sys.path.
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from lm_student.lm_student_env import LMStudentEnv
from lm_student.tiny_lm_head import GPT2FeatureExtractor


def evaluate_lm_model(
    model: PPO,
    tokenizer: AutoTokenizer,
    episodes: int = 30,
) -> np.ndarray:
    """Evaluate across all task_id/difficulty combinations."""
    acc_matrix = np.zeros((5, 3), dtype=float)
    for task_id in range(5):
        for diff in range(3):
            env = LMStudentEnv(tokenizer, task_id=task_id, difficulty=diff)
            correct = 0
            for _ in range(episodes):
                obs, _ = env.reset()
                action, _ = model.predict(obs, deterministic=True)
                _, reward, *_ = env.step(int(action))
                correct += reward
            acc_matrix[task_id, diff] = correct / episodes
    return acc_matrix


def plot_heatmap(acc_matrix: np.ndarray, path: str) -> None:
    plt.figure(figsize=(6, 5))
    plt.imshow(acc_matrix, vmin=0, vmax=1, cmap="viridis")
    plt.colorbar(label="accuracy")
    plt.xticks(range(3), [0, 1, 2])
    plt.yticks(range(5), [f"task {i}" for i in range(5)])
    plt.xlabel("difficulty")
    plt.ylabel("task")
    plt.title("LM student accuracy")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def train_lm_student(
    total_timesteps: int = 2000,
    max_length: int = 64,
    model_name: str = "gpt2",
    save_path: str = "models/lm_student_ppo.zip",
) -> None:
    """Train PPO with a frozen GPT-2 features extractor."""
    torch.set_num_threads(1)
    os.makedirs("models", exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    env = LMStudentEnv(tokenizer, task_id=0, difficulty=0, max_length=max_length)
    policy_kwargs = {
        "features_extractor_class": GPT2FeatureExtractor,
        "features_extractor_kwargs": {"model_name": model_name, "max_length": max_length},
        "net_arch": [64, 64],
    }

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        n_steps=64,
        batch_size=32,
        policy_kwargs=policy_kwargs,
    )
    model.learn(total_timesteps=total_timesteps)
    model.save(save_path)
    print(f"Saved LM student to {save_path}")

    # Evaluate across all tasks/difficulties.
    acc_matrix = evaluate_lm_model(model, tokenizer, episodes=20)
    heatmap_path = "models/lm_student_accuracy.png"
    plot_heatmap(acc_matrix, heatmap_path)
    print(f"Saved LM student accuracy heatmap to {heatmap_path}")


if __name__ == "__main__":
    train_lm_student()
