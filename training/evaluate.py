"""Evaluation utilities for trained PPO agents (multi-step aware, no leakage)."""

import os
import sys
from typing import Optional

import torch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from stable_baselines3 import PPO

from student.student_env import StudentEnv


def evaluate_model(
    model,
    episodes: int = 200,
    family_id: int = 0,
    difficulty_id: int = 0,
    env: Optional[StudentEnv] = None,
) -> float:
    """Run full multi-step episodes; success only if final action is correct."""
    eval_env = env or StudentEnv(family_id=family_id, difficulty_id=difficulty_id)
    successes = 0
    max_steps = 20  # hard safety cap

    for _ in range(episodes):
        obs, info = eval_env.reset()
        done = False
        steps = 0
        final_correct = False

        while not done and steps < max_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = eval_env.step(action)
            steps += 1

        if steps >= max_steps:
            # This warns about a pathological stuck episode
            print("[WARN] Episode exceeded max_steps")

        # Only count an episode if it terminated properly and was correct
        if done and info.get("correct"):
            successes += 1

    return successes / episodes



def evaluate_saved_model(
    model_path: str = "models/ppo_task0.zip",
    episodes: int = 20,
    family_id: int = 0,
    difficulty_id: int = 0,
) -> float:
    """Load a saved PPO model and report accuracy."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")

    env = StudentEnv(family_id=family_id, difficulty_id=difficulty_id)
    model = PPO.load(model_path, env=env)
    accuracy = evaluate_model(
        model,
        episodes=episodes,
        family_id=family_id,
        difficulty_id=difficulty_id,
        env=env,
    )
    print(f"Accuracy over {episodes} episodes: {accuracy * 100:.2f}%")
    return accuracy


if __name__ == "__main__":
    evaluate_saved_model()
