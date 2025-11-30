"""Streamlit demo for interacting with the trained student agent."""

import os
import sys
from typing import List, Optional, Tuple

import numpy as np
import streamlit as st
from stable_baselines3 import PPO

# Ensure repository root is on sys.path for direct script execution.
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from tasks.task_generator import generate_task


def main() -> None:
    st.title("MentorFlow Student Demo")

    default_model_paths = [
        "models/ppo_teacher_curriculum.zip",
        "models/ppo_task0.zip",
        "models/ppo_task0_logged.zip",
    ]
    existing_paths = [p for p in default_model_paths if os.path.exists(p)]
    model_path = st.sidebar.selectbox("Model path", options=existing_paths)
    model = load_model(model_path)

    task_id = st.sidebar.selectbox("Task family (0-4)", options=list(range(5)), index=0)
    difficulty = st.sidebar.selectbox("Difficulty (0-2)", options=[0, 1, 2], index=0)
    batch_size = st.sidebar.number_input("Batch eval size", min_value=1, max_value=200, value=50, step=1)

    if st.button("Sample one task"):
        task = generate_task(task_id, difficulty, seed=None)
        obs = np.asarray(task.obs_vec, dtype=np.float32)
        st.write("Observation vector:", obs)
        st.write("Choices:")
        for i, c in enumerate(task.human_choices):
            st.write(f"{i}: {c}")
        action, _ = model.predict(obs, deterministic=True)
        st.write(f"Student predicted: {int(action)} ({task.human_choices[int(action)]})")
        st.write(f"Correct: {task.correct_action} ({task.human_choices[task.correct_action]})")
        st.success("Correct!" if int(action) == task.correct_action else "Incorrect")

    if st.button("Batch evaluate"):
        correct = 0
        for _ in range(batch_size):
            task = generate_task(task_id, difficulty, seed=None)
            obs = np.asarray(task.obs_vec, dtype=np.float32)
            action, _ = model.predict(obs, deterministic=True)
            correct += 1 if int(action) == task.correct_action else 0
        acc = correct / batch_size
        st.write(f"Accuracy over {batch_size} samples: {acc*100:.2f}%")

    # Show existing heatmap if available.
    heatmap_paths = [
        "models/teacher_curriculum_heatmap.png",
        "demo_accuracy_overview.png",
    ]
    for hp in heatmap_paths:
        if os.path.exists(hp):
            st.subheader(f"Heatmap: {hp}")
            st.image(hp)


if __name__ == "__main__":
    main()
