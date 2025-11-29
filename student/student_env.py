"""
Student Environment for TeachRL
Gym-compatible environment where the student learns to solve coding microtasks.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tasks.task_generator import (
    TaskSpec, generate_task, generate_task_by_arm,
    OBS_VEC_SIZE, NUM_CHOICES, NUM_FAMILIES, NUM_DIFFICULTIES
)


class StudentEnv(gym.Env):
    """
    Gym environment for the coding task student.
    
    Observation: Fixed-length vector encoding the task
    Action: Choice index (0-3)
    Reward: +1 for correct, 0 for incorrect
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(
        self,
        family_id: Optional[int] = None,
        difficulty_id: Optional[int] = None,
        arm_id: Optional[int] = None,
        seed: Optional[int] = None,
        render_mode: Optional[str] = None
    ):
        """
        Initialize environment.
        
        Args:
            family_id: Fixed family (0-4), or None for random
            difficulty_id: Fixed difficulty (0-2), or None for random
            arm_id: Alternative to family_id/difficulty_id (0-14)
            seed: RNG seed for task generation
            render_mode: "human" for printing tasks
        """
        super().__init__()
        
        # Task selection mode
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
        
        # Spaces
        self.observation_space = spaces.Box(
            low=-100.0,
            high=100.0,
            shape=(OBS_VEC_SIZE,),
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(NUM_CHOICES)
        
        # State
        self.current_task: Optional[TaskSpec] = None
        self.task_seed = seed
        self.render_mode = render_mode
        self._rng = np.random.default_rng(seed)
        
        # Episode tracking
        self.episode_count = 0
        self.correct_count = 0
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset environment with a new task.
        
        Args:
            seed: Optional seed override
            options: Can contain 'family_id', 'difficulty_id', or 'arm_id'
        
        Returns:
            observation, info dict
        """
        super().reset(seed=seed)
        
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        
        # Determine task type
        family_id = self.family_id
        difficulty_id = self.difficulty_id
        
        if options:
            if 'arm_id' in options:
                arm_id = options['arm_id']
                family_id = arm_id // NUM_DIFFICULTIES
                difficulty_id = arm_id % NUM_DIFFICULTIES
            else:
                family_id = options.get('family_id', family_id)
                difficulty_id = options.get('difficulty_id', difficulty_id)
        
        # Random selection if not fixed
        if family_id is None:
            family_id = int(self._rng.integers(0, NUM_FAMILIES))
        if difficulty_id is None:
            difficulty_id = int(self._rng.integers(0, NUM_DIFFICULTIES))
        
        # Generate task
        task_seed = int(self._rng.integers(0, 2**31 - 1))
        self.current_task = generate_task(family_id, difficulty_id, task_seed)
        
        self.episode_count += 1
        
        obs = np.array(self.current_task.obs_vec, dtype=np.float32)
        info = {
            "family_id": self.current_task.family_id,
            "difficulty_id": self.current_task.difficulty_id,
            "human_prompt": self.current_task.human_prompt,
            "human_choices": self.current_task.human_choices,
        }
        
        if self.render_mode == "human":
            self._render_task()
        
        return obs, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Take action (select a choice).
        
        Args:
            action: Choice index (0-3)
        
        Returns:
            observation, reward, terminated, truncated, info
        """
        if self.current_task is None:
            raise RuntimeError("Must call reset() before step()")
        
        # Check correctness
        correct = (action == self.current_task.correct_action)
        reward = 1.0 if correct else 0.0
        
        if correct:
            self.correct_count += 1
        
        # Episode ends after one step (single-step task)
        terminated = True
        truncated = False
        
        info = {
            "correct": correct,
            "correct_action": self.current_task.correct_action,
            "chosen_action": action,
            "family_id": self.current_task.family_id,
            "difficulty_id": self.current_task.difficulty_id,
        }
        
        if self.render_mode == "human":
            self._render_result(action, correct)
        
        # Return same observation (episode is done anyway)
        obs = np.array(self.current_task.obs_vec, dtype=np.float32)
        
        return obs, reward, terminated, truncated, info
    
    def _render_task(self):
        """Print task to console."""
        print(f"\n{'='*50}")
        print(f"Task [{self.current_task.family_id}][{self.current_task.difficulty_id}]")
        print(f"{'='*50}")
        print(self.current_task.human_prompt)
        print(f"\nChoices:")
        for i, choice in enumerate(self.current_task.human_choices):
            print(f"  [{i}] {choice}")
    
    def _render_result(self, action: int, correct: bool):
        """Print result to console."""
        result = "✓ CORRECT" if correct else "✗ WRONG"
        print(f"\nYour answer: [{action}] {self.current_task.human_choices[action]}")
        print(f"Result: {result}")
        if not correct:
            ca = self.current_task.correct_action
            print(f"Correct was: [{ca}] {self.current_task.human_choices[ca]}")
    
    def get_accuracy(self) -> float:
        """Get accuracy over all episodes."""
        if self.episode_count == 0:
            return 0.0
        return self.correct_count / self.episode_count
    
    def reset_stats(self):
        """Reset accuracy tracking."""
        self.episode_count = 0
        self.correct_count = 0


class MultiTaskStudentEnv(StudentEnv):
    """
    Environment that samples tasks from all families/difficulties.
    Used for general training or evaluation.
    """
    
    def __init__(self, seed: Optional[int] = None, render_mode: Optional[str] = None):
        super().__init__(
            family_id=None,
            difficulty_id=None,
            seed=seed,
            render_mode=render_mode
        )


class SingleTaskEnv(StudentEnv):
    """
    Environment fixed to a single task type (for curriculum training).
    """
    
    def __init__(
        self,
        arm_id: int,
        seed: Optional[int] = None,
        render_mode: Optional[str] = None
    ):
        super().__init__(
            arm_id=arm_id,
            seed=seed,
            render_mode=render_mode
        )


def make_env(arm_id: Optional[int] = None, seed: Optional[int] = None):
    """
    Factory function to create environment.
    
    Args:
        arm_id: If provided, creates single-task env. Otherwise multi-task.
        seed: RNG seed
    
    Returns:
        Gym environment
    """
    if arm_id is not None:
        return SingleTaskEnv(arm_id=arm_id, seed=seed)
    return MultiTaskStudentEnv(seed=seed)


# =============================================================================
# STANDALONE TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("TeachRL Student Environment Test")
    print("=" * 60)
    
    # Test multi-task environment
    print("\n--- Multi-Task Environment ---")
    env = MultiTaskStudentEnv(seed=42, render_mode="human")
    
    obs, info = env.reset()
    print(f"\nObs shape: {obs.shape}")
    print(f"Obs sample: {obs[:10]}...")
    
    # Take random action
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"\nReward: {reward}, Terminated: {terminated}")
    
    # Test single-task environment
    print("\n\n--- Single-Task Environment (arm_id=7) ---")
    env2 = SingleTaskEnv(arm_id=7, seed=123, render_mode="human")
    
    obs, info = env2.reset()
    action = env2.action_space.sample()
    obs, reward, terminated, truncated, info = env2.step(action)
    
    # Test accuracy tracking
    print("\n\n--- Accuracy Test (100 random episodes) ---")
    env3 = MultiTaskStudentEnv(seed=0)
    for _ in range(100):
        obs, _ = env3.reset()
        action = env3.action_space.sample()
        env3.step(action)
    
    print(f"Random policy accuracy: {env3.get_accuracy():.2%}")
    print(f"(Expected ~25% for 4 choices)")
    
    # Test gym compatibility
    print("\n--- Gym Compatibility Check ---")
    from gymnasium.utils.env_checker import check_env
    env4 = MultiTaskStudentEnv(seed=42)
    try:
        check_env(env4, skip_render_check=True)
        print("✓ Environment passes Gym checks!")
    except Exception as e:
        print(f"✗ Gym check failed: {e}")
