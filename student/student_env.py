"""Multi-step student environment with no answer leakage."""

import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.utils import seeding

sys.path.insert(0, str(Path(__file__).parent.parent))
from tasks.task_generator import (
    TaskSpec,
    generate_task,
    generate_task_by_arm,
    OBS_VEC_SIZE,
    NUM_CHOICES,
    NUM_FAMILIES,
    NUM_DIFFICULTIES,
)


class StudentEnv(gym.Env):
    """Multi-step env: intermediate observations, reward only on final step."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        family_id: Optional[int] = None,
        difficulty_id: Optional[int] = None,
        arm_id: Optional[int] = None,
        seed: Optional[int] = None,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        if arm_id is not None:
            self.family_id = arm_id // NUM_DIFFICULTIES
            self.difficulty_id = arm_id % NUM_DIFFICULTIES
            self.fixed_task_type = True
        elif family_id is not None and difficulty_id is not None:
            self.family_id = family_id
            self.difficulty_id = difficulty_id
            self.fixed_task_type = True
        else:
            self.family_id = None
            self.difficulty_id = None
            self.fixed_task_type = False

        self.observation_space = spaces.Box(
            low=-100.0, high=100.0, shape=(OBS_VEC_SIZE,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(NUM_CHOICES)

        self.current_task: Optional[TaskSpec] = None
        self.current_task_steps: List[np.ndarray] = []
        self.current_step = 0
        self.task_seed = seed
        self.render_mode = render_mode
        self._rng = np.random.default_rng(seed)

    def _mask_obs(self, obs: np.ndarray, step_idx: int, total_steps: int) -> np.ndarray:
        """Gradually reveal the observation; ensure no leakage of answer index."""
        masked = obs.copy()
        # Random mask ratio per step to avoid deterministic mapping.
        reveal_ratio = (step_idx + 1) / total_steps
        mask_start = int(reveal_ratio * len(masked))
        masked[mask_start:] = 0.0
        return masked

    def _build_steps(self, task: TaskSpec) -> List[np.ndarray]:
        total_steps = int(self._rng.integers(3, 7))
        obs = np.array(task.obs_vec, dtype=np.float32)
        return [self._mask_obs(obs, i, total_steps) for i in range(total_steps)]

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        family_id = self.family_id
        difficulty_id = self.difficulty_id
        if options:
            if "arm_id" in options:
                arm_id = options["arm_id"]
                family_id = arm_id // NUM_DIFFICULTIES
                difficulty_id = arm_id % NUM_DIFFICULTIES
            else:
                family_id = options.get("family_id", family_id)
                difficulty_id = options.get("difficulty_id", difficulty_id)

        if family_id is None:
            family_id = int(self._rng.integers(0, NUM_FAMILIES))
        if difficulty_id is None:
            difficulty_id = int(self._rng.integers(0, NUM_DIFFICULTIES))

        task_seed = int(self._rng.integers(0, 2**31 - 1))
        self.current_task = generate_task(family_id, difficulty_id, task_seed)
        self.current_task_steps = self._build_steps(self.current_task)
        self.current_step = 0

        obs = self.current_task_steps[self.current_step]
        info = {
            "family_id": self.current_task.family_id,
            "difficulty_id": self.current_task.difficulty_id,
            "step": self.current_step,
            "is_final_step": False,
            "correct_final_action": self.current_task.correct_action,
        }
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        if self.current_task is None:
            raise RuntimeError("Must call reset() before step()")

        total_steps = len(self.current_task_steps)
        is_final = self.current_step == total_steps - 1

        reward = 0.0
        correct = False
        if is_final:
            correct = (action == self.current_task.correct_action)
            reward = 1.0 if correct else 0.0

        self.current_step += 1
        terminated = self.current_step >= total_steps
        truncated = False

        if terminated:
            next_obs = np.zeros_like(self.current_task_steps[-1])
        else:
            next_obs = self.current_task_steps[self.current_step]

        info = {
            "correct": correct,
            "correct_action": self.current_task.correct_action,
            "chosen_action": action,
            "family_id": self.current_task.family_id,
            "difficulty_id": self.current_task.difficulty_id,
            "step": self.current_step,
            "is_final_step": terminated,
            "correct_final_action": self.current_task.correct_action,
        }
        return next_obs, reward, terminated, truncated, info
