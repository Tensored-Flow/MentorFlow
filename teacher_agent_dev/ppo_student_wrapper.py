"""
PPO Student wrapper that implements StudentAgentInterface for use in compare_strategies.py.
Adapts PPO agents to work with MockTaskGenerator's topic/difficulty strings.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import math
import logging

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm

from student.student_env import StudentEnv
from tasks.task_generator import NUM_FAMILIES, NUM_DIFFICULTIES
from training.evaluate import evaluate_model

from interfaces import StudentAgentInterface, Task, StudentState


# Map MockTaskGenerator topics to family_ids (use first NUM_FAMILIES topics)
def map_topic_to_family_id(topic: str, available_topics: List[str]) -> int:
    """Map topic string to family_id (0 to NUM_FAMILIES-1)."""
    try:
        topic_idx = available_topics.index(topic)
        return topic_idx % NUM_FAMILIES  # Wrap around if more topics than families
    except ValueError:
        # Fallback: hash-based mapping
        return hash(topic) % NUM_FAMILIES


def map_difficulty_to_difficulty_id(difficulty: str, available_difficulties: List[str]) -> int:
    """Map difficulty string to difficulty_id (0 to NUM_DIFFICULTIES-1)."""
    try:
        diff_idx = available_difficulties.index(difficulty)
        # Map to 0-4 range
        return min(diff_idx, NUM_DIFFICULTIES - 1)
    except ValueError:
        # Fallback: map by name
        difficulty_map = {
            'trivial': 0, 'easy': 1, 'medium': 2, 'hard': 3, 'expert': 4,
            'master': 3, 'grandmaster': 4  # Map higher difficulties to top levels
        }
        return difficulty_map.get(difficulty.lower(), 2)  # Default to medium


class PPOStudentWrapper(StudentAgentInterface):
    """
    Wrapper around PPO agent to match StudentAgentInterface.
    Uses a single PPO model that trains on different task types.
    """
    
    def __init__(
        self,
        learning_rate: float = 3e-4,
        train_steps_per_task: int = 64,  # Steps to train per task (balance speed/variance)
        seed: Optional[int] = None,
        available_topics: Optional[List[str]] = None,
        available_difficulties: Optional[List[str]] = None,
        forget_rate: float = 0.05,  # Faster forgetting to avoid flat curves
    ):
        """
        Initialize PPO student wrapper.
        
        Args:
            learning_rate: PPO learning rate
            train_steps_per_task: Number of training steps per task
            seed: Random seed
            available_topics: List of available topics (for mapping)
            available_difficulties: List of available difficulties (for mapping)
        """
        self.train_steps_per_task = train_steps_per_task
        self.seed = seed
        self.available_topics = available_topics or []
        self.available_difficulties = available_difficulties or []
        self.forget_rate = forget_rate
        
        # Initialize PPO model with a default environment
        # Will be updated per task during training
        import torch
        import os
        # Set single thread to avoid mutex issues on macOS
        torch.set_num_threads(1)
        os.environ['OMP_NUM_THREADS'] = '1'
        
        default_env = StudentEnv(family_id=0, difficulty_id=0, seed=seed)
        self.model: Optional[BaseAlgorithm] = PPO(
            "MlpPolicy",
            default_env,
            learning_rate=learning_rate,
            n_steps=64,  # Larger rollout for more stable updates
            batch_size=64,
            n_epochs=4,
            gamma=0.95,
            gae_lambda=0.92,
            clip_range=0.2,
            ent_coef=0.05,  # more exploration to reduce linearity
            verbose=0,
            seed=seed,
        )
        
        # Track statistics for state
        self.topic_attempts: Dict[str, int] = {}
        self.topic_correct: Dict[str, int] = {}
        self.topic_base_skills: Dict[str, float] = {}
        self.total_time: float = 0.0
    
    def _get_family_id(self, topic: str) -> int:
        """Get family_id for a topic."""
        return map_topic_to_family_id(topic, self.available_topics)
    
    def _get_difficulty_id(self, difficulty: str) -> int:
        """Get difficulty_id for a difficulty."""
        return map_difficulty_to_difficulty_id(difficulty, self.available_difficulties)
    
    def learn(self, task: Task) -> bool:
        """
        Train PPO model on the given task.
        
        Args:
            task: Task to learn from
            
        Returns:
            True if model would have been correct (approximate via evaluation)
        """
        family_id = self._get_family_id(task.topic)
        difficulty_id = self._get_difficulty_id(task.difficulty)
        
        # Create environment for this task type
        env = StudentEnv(family_id=family_id, difficulty_id=difficulty_id, seed=self.seed)
        self.model.set_env(env)
        
        # Train PPO for a few steps on this task type
        # PPO needs at least n_steps (32) to collect a rollout before learning
        # But we can accumulate steps across multiple calls, so allow smaller values
        # Use n_steps as minimum only for first few calls
        actual_steps = max(self.train_steps_per_task, self.model.n_steps)
        try:
            self.model.learn(total_timesteps=actual_steps, reset_num_timesteps=False, progress_bar=False)
        except Exception as e:
            logging.warning(f"PPO learn failed on task {task.topic}/{task.difficulty}: {e}")
            # Continue but signal we couldn't learn
        
        # Quick correctness check: small rollout to approximate correctness
        try:
            obs, _ = env.reset()
            done = False
            steps = 0
            while not done and steps < 20:
                # Stochastic predict to inject variance
                action, _ = self.model.predict(obs, deterministic=False)
                obs, reward, done, truncated, info = env.step(action)
                steps += 1
            was_correct = info.get("correct", False) if done else False
        except:
            was_correct = False  # Default to incorrect if evaluation fails
        
        # Update statistics
        self.topic_attempts[task.topic] = self.topic_attempts.get(task.topic, 0) + 1
        # Track correct counts even when first attempt is wrong to avoid KeyError
        self.topic_correct[task.topic] = self.topic_correct.get(task.topic, 0) + (1 if was_correct else 0)
        
        base_skill = self.topic_correct[task.topic] / self.topic_attempts[task.topic]
        self.topic_base_skills[task.topic] = base_skill
        
        return was_correct
    
    def evaluate(self, eval_tasks: List[Task]) -> float:
        """
        Evaluate model on a list of tasks.
        
        Args:
            eval_tasks: List of tasks to evaluate on
            
        Returns:
            Average accuracy (0.0 to 1.0)
        """
        if not eval_tasks:
            return 0.0
        
        accuracies = []
        
        # Group tasks by (family_id, difficulty_id) to avoid recreating envs
        task_groups: Dict[tuple, List[Task]] = {}
        for task in eval_tasks:
            family_id = self._get_family_id(task.topic)
            difficulty_id = self._get_difficulty_id(task.difficulty)
            key = (family_id, difficulty_id)
            if key not in task_groups:
                task_groups[key] = []
            task_groups[key].append(task)
        
        # Evaluate on each group
        for (family_id, difficulty_id), group_tasks in task_groups.items():
            env = StudentEnv(family_id=family_id, difficulty_id=difficulty_id, seed=self.seed)
            # Evaluate with a few more episodes for stability
            accuracy = evaluate_model(
                self.model,
                episodes=max(8, len(group_tasks)),  # more episodes -> less linear
                family_id=family_id,
                difficulty_id=difficulty_id,
                env=env,
            )
            accuracies.append(accuracy)
        
        # Return average accuracy across all task groups
        return float(np.mean(accuracies)) if accuracies else 0.0
    
    def get_state(self) -> StudentState:
        """Get current student state."""
        # Map topic_base_skills to topic_accuracies format
        topic_accuracies = {topic: skill for topic, skill in self.topic_base_skills.items()}
        topic_attempts_dict = {topic: self.topic_attempts.get(topic, 0) for topic in self.topic_base_skills.keys()}
        
        # Create time_since_practice dict (simplified - all topics have same time)
        time_since_practice = {topic: self.total_time for topic in self.topic_base_skills.keys()}
        
        return StudentState(
            topic_accuracies=topic_accuracies,
            topic_attempts=topic_attempts_dict,
            time_since_practice=time_since_practice,
            total_timesteps=int(self.total_time * self.train_steps_per_task),  # Approximate
            current_time=self.total_time
        )
    
    def answer(self, task: Task) -> int:
        """
        Answer a task using the PPO model.
        
        Args:
            task: Task to answer
            
        Returns:
            Index of chosen answer (0-3)
        """
        family_id = self._get_family_id(task.topic)
        difficulty_id = self._get_difficulty_id(task.difficulty)
        
        # Create environment for this task type
        env = StudentEnv(family_id=family_id, difficulty_id=difficulty_id, seed=self.seed)
        
        # Reset environment to get observation
        obs, info = env.reset()
        
        # Predict action (answer)
        action, _ = self.model.predict(obs, deterministic=True)
        
        return int(action)
    
    def advance_time(self, delta: float) -> None:
        """Advance time and decay learned skills to simulate forgetting."""
        self.total_time += delta
        if self.forget_rate > 0:
            decay = math.exp(-self.forget_rate * delta)
            for topic in list(self.topic_base_skills.keys()):
                self.topic_base_skills[topic] *= decay
