"""Realistic mock student agent with learning and forgetting capabilities."""

import random
from typing import Dict, List
import numpy as np
from interfaces import Task, StudentState, StudentAgentInterface


class MockStudentAgent(StudentAgentInterface):
    """
    Realistic mock student with:
    - Learning: improves with practice
    - Forgetting: Ebbinghaus curve
    - Per-topic skill tracking
    """
    
    def __init__(self, learning_rate: float = 0.15, forgetting_rate: float = 0.05, seed: int = 42):
        """
        Initialize mock student.
        
        Args:
            learning_rate: How much skill improves per practice (0-1)
            forgetting_rate: How fast retention decays (higher = faster forgetting)
            seed: Random seed for reproducibility
        """
        self.learning_rate = learning_rate
        self.forgetting_rate = forgetting_rate
        self.rng = random.Random(seed)
        
        # Track per-topic base skill (0.0 to 1.0)
        # Starts at 0.0 = random guessing (25% accuracy on 4-choice MCQ)
        self.topic_skills: Dict[str, float] = {}
        
        # Track history
        self.topic_attempts: Dict[str, int] = {}
        self.last_practice_time: Dict[str, float] = {}
        
        # Time tracking for forgetting simulation
        self.current_time = 0.0
        self.total_timesteps = 0
        
        # Difficulty learning factors (harder = slower learning)
        self.difficulty_factors = {
            'easy': 1.0,
            'medium': 0.7,
            'hard': 0.5
        }
    
    def answer(self, task: Task) -> int:
        """
        Answer a task based on effective skill (accounting for forgetting).
        
        Returns:
            Index of chosen answer (0-3)
        """
        effective_skill = self._get_effective_skill(task.topic)
        
        # Probability of correct = 0.25 (random) + 0.75 * effective_skill
        # If skill=0.0: 25% accuracy (random guessing)
        # If skill=1.0: 100% accuracy (perfect)
        prob_correct = 0.25 + 0.75 * effective_skill
        
        if self.rng.random() < prob_correct:
            # Answer correctly
            return task.answer
        else:
            # Answer incorrectly - choose random wrong answer
            wrong_answers = [i for i in range(4) if i != task.answer]
            return self.rng.choice(wrong_answers)
    
    def learn(self, task: Task) -> bool:
        """
        Learn from a task.
        
        Learning formula:
        new_skill = current_skill + learning_rate * difficulty_factor * (1 - current_skill)
        
        This means:
        - Easy tasks help more (difficulty_factor = 1.0)
        - Hard tasks help less (difficulty_factor = 0.5)
        - Learning slows as skill approaches 1.0
        
        Returns:
            Whether answer was correct
        """
        # Answer the task first
        was_correct = (self.answer(task) == task.answer)
        
        # Update skill based on practice
        topic = task.topic
        difficulty = task.difficulty
        
        # Get current base skill (before forgetting)
        if topic not in self.topic_skills:
            self.topic_skills[topic] = 0.0
            self.topic_attempts[topic] = 0
            self.last_practice_time[topic] = self.current_time
        
        current_base_skill = self.topic_skills[topic]
        difficulty_factor = self.difficulty_factors[difficulty]
        
        # Update skill: learning is faster for easy tasks, slower as skill increases
        # If correct, learn more; if wrong, learn less (but still learn)
        if was_correct:
            learning_multiplier = 1.0
        else:
            learning_multiplier = 0.3  # Learn less from mistakes
        
        skill_increase = (
            self.learning_rate * 
            difficulty_factor * 
            learning_multiplier * 
            (1.0 - current_base_skill)
        )
        
        self.topic_skills[topic] = min(1.0, current_base_skill + skill_increase)
        self.topic_attempts[topic] = self.topic_attempts.get(topic, 0) + 1
        self.last_practice_time[topic] = self.current_time
        self.total_timesteps += 1
        
        return was_correct
    
    def _get_effective_skill(self, topic: str) -> float:
        """
        Get effective skill accounting for forgetting (Ebbinghaus curve).
        
        Formula: effective_skill = base_skill * retention
        retention = exp(-forgetting_rate * time_since_practice)
        """
        if topic not in self.topic_skills:
            return 0.0
        
        base_skill = self.topic_skills[topic]
        time_since = self.current_time - self.last_practice_time.get(topic, self.current_time)
        
        # Ebbinghaus forgetting curve
        retention = np.exp(-self.forgetting_rate * time_since)
        
        # Effective skill = base skill reduced by forgetting
        effective_skill = base_skill * retention
        
        return max(0.0, min(1.0, effective_skill))
    
    def evaluate(self, eval_tasks: List[Task]) -> float:
        """
        Evaluate student on a list of tasks.
        
        Returns:
            Accuracy (0.0 to 1.0)
        """
        if not eval_tasks:
            return 0.0
        
        correct = 0
        for task in eval_tasks:
            answer = self.answer(task)
            if answer == task.answer:
                correct += 1
        
        return correct / len(eval_tasks)
    
    def get_state(self) -> StudentState:
        """Get current student state."""
        # Calculate accuracies for each topic (simulated, not actual eval)
        topic_accuracies = {}
        for topic in self.topic_skills.keys():
            effective_skill = self._get_effective_skill(topic)
            # Convert skill to accuracy: 0.25 (random) + 0.75 * skill
            topic_accuracies[topic] = 0.25 + 0.75 * effective_skill
        
        # Calculate time since practice for each topic
        time_since_practice = {}
        for topic in self.last_practice_time:
            time_since_practice[topic] = self.current_time - self.last_practice_time[topic]
        
        return StudentState(
            topic_accuracies=topic_accuracies,
            topic_attempts=self.topic_attempts.copy(),
            time_since_practice=time_since_practice,
            total_timesteps=self.total_timesteps,
            current_time=self.current_time
        )
    
    def advance_time(self, delta: float = 1.0):
        """Advance time for forgetting simulation."""
        self.current_time += delta

