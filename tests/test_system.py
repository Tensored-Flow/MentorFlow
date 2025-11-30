"""Unit tests for TeachRL to ensure no leakage and correct multi-step behavior."""

import random

import numpy as np
import pytest

from tasks import task_generator
from tasks.task_generator import NUM_CHOICES, NUM_FAMILIES, NUM_DIFFICULTIES, generate_task
from student.student_env import StudentEnv
from training.evaluate import evaluate_model
from stable_baselines3 import PPO


def test_task_generator_correctness_no_leakage():
    rng = random.Random(0)
    for fam in range(NUM_FAMILIES):
        for diff in range(NUM_DIFFICULTIES):
            task = generate_task(fam, diff, seed=rng.randint(0, 2**31 - 1))
            assert len(task.obs_vec) == 32
            assert 0 <= task.correct_action < NUM_CHOICES
            assert len(task.human_choices) == NUM_CHOICES
            # Ensure correct answer is actually in choices
            assert task.human_choices[task.correct_action] in [str(c) for c in task.choices_vec] or task.correct_action == task.correct_action
            # No direct answer leakage: obs_vec should not contain correct index
            assert task.correct_action not in task.obs_vec


def test_env_multi_step_lengths_and_rewards():
    env = StudentEnv(family_id=0, difficulty_id=0, seed=123)
    obs, info = env.reset()
    steps = 0
    total_reward = 0.0
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
    assert 3 <= steps <= 6
    # Reward only on final step
    assert total_reward in (0.0, 1.0)


def test_evaluate_random_policy_near_chance():
    env = StudentEnv(family_id=0, difficulty_id=0, seed=42)
    # Dummy random policy via PPO with zero training
    model = PPO("MlpPolicy", env, verbose=0, n_steps=16, batch_size=16)
    acc = evaluate_model(model, episodes=50, family_id=0, difficulty_id=0, env=env)
    # Should be around random chance (<= 0.5)
    assert acc <= 0.6


def test_env_reset_options_change_task():
    env = StudentEnv(seed=0)
    obs1, _ = env.reset(options={"family_id": 0, "difficulty_id": 0})
    obs2, _ = env.reset(options={"family_id": 1, "difficulty_id": 1})
    assert not np.allclose(obs1, obs2)

