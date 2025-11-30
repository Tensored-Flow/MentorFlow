"""Enhanced mock student agent with PPO-like features: transfer learning, exponential learning curves."""

import random
from typing import Dict, List, Set, Optional
import numpy as np
from interfaces import Task, StudentState, StudentAgentInterface


class MockStudentAgent(StudentAgentInterface):
    """
    Enhanced mock student with PPO-like features:
    - Learning: improves with practice (exponential when guided, linear when random)
    - Forgetting: Ebbinghaus curve
    - Per-topic skill tracking
    - Transfer learning: skills in related topics help each other
    - Feature representations: abstract concepts that transfer across topics
    - Exponential learning curve when teacher-guided (coherent curriculum)
    - Stochastic/erratic learning when random
    """
    
    def __init__(
        self, 
        learning_rate: float = 0.15, 
        forgetting_rate: float = 0.01,  # Reduced for long training
        transfer_strength: float = 0.3,  # How much skills transfer between topics
        seed: int = 42,
        curriculum_coherence: Optional[float] = None  # Track if teacher-guided
    ):
        """
        Initialize enhanced mock student.
        
        Args:
            learning_rate: Base learning rate (0-1)
            forgetting_rate: How fast retention decays
            transfer_strength: How much skills transfer (0-1)
            seed: Random seed
            curriculum_coherence: Track if following coherent curriculum (auto-detected)
        """
        self.learning_rate = learning_rate
        self.forgetting_rate = forgetting_rate
        self.transfer_strength = transfer_strength
        self.rng = random.Random(seed)
        
        # Track per-topic base skill (0.0 to 1.0)
        self.topic_skills: Dict[str, float] = {}
        
        # PPO-like: Feature representations (abstract concepts that transfer)
        # Groups of related topics share feature representations
        self.feature_representations: Dict[str, Set[str]] = self._build_feature_groups()
        
        # Track history
        self.topic_attempts: Dict[str, int] = {}
        self.last_practice_time: Dict[str, float] = {}
        
        # Time tracking for forgetting simulation
        self.current_time = 0.0
        self.total_timesteps = 0
        
        # Track curriculum coherence (exponential learning vs stochastic)
        self.curriculum_coherence = curriculum_coherence
        self.recent_topics: List[str] = []  # Track recent topic sequence
        self.recent_topics_window = 5
        
        # Expanded difficulty learning factors (all 7 levels)
        self.difficulty_factors = {
            'trivial': 1.2,      # Very easy, learn quickly
            'easy': 1.0,         # Standard easy
            'medium': 0.8,       # Moderate
            'hard': 0.6,         # Challenging
            'expert': 0.4,       # Very hard (multi-step)
            'master': 0.25,      # Extremely hard
            'grandmaster': 0.15  # Maximum difficulty
        }
        
        # Multi-step penalty: harder difficulties need more practice
        self.multi_step_penalty = {
            'trivial': 0.0,
            'easy': 0.0,
            'medium': 0.1,
            'hard': 0.2,
            'expert': 0.3,
            'master': 0.4,
            'grandmaster': 0.5
        }
    
    def _build_feature_groups(self) -> Dict[str, Set[str]]:
        """Build groups of related topics for transfer learning."""
        # Group related topics that share underlying concepts
        return {
            'stem_concepts': {'mathematics', 'programming', 'science', 'physics', 'chemistry'},
            'humanities_concepts': {'history', 'literature', 'philosophy', 'art'},
            'social_concepts': {'current_events', 'economics', 'psychology', 'geography'},
            'abstract_reasoning': {'mathematics', 'programming', 'philosophy'},
            'memorization': {'history', 'geography', 'biology', 'chemistry'}
        }
    
    def _get_transfer_boost(self, topic: str) -> float:
        """
        Calculate transfer learning boost from related topics.
        
        Returns:
            Multiplier for learning rate based on related topic skills
        """
        boost = 0.0
        
        # Find which feature groups this topic belongs to
        for feature_name, topics in self.feature_representations.items():
            if topic in topics:
                # Get average skill from related topics
                related_skills = [
                    self.topic_skills.get(t, 0.0)
                    for t in topics
                    if t != topic and t in self.topic_skills
                ]
                if related_skills:
                    avg_related_skill = np.mean(related_skills)
                    # Transfer boost proportional to related skills
                    boost += self.transfer_strength * avg_related_skill * 0.5
        
        return min(boost, 0.5)  # Cap at 50% boost
    
    def _get_curriculum_coherence(self) -> float:
        """
        Detect if student is following coherent curriculum (teacher-guided).
        
        Returns:
            Coherence score (0.0 = random, 1.0 = very coherent)
        """
        if len(self.recent_topics) < 3:
            return 0.5  # Neutral
        
        # Check if topics are related (same feature groups)
        recent_set = set(self.recent_topics[-3:])
        coherence_score = 0.0
        
        for feature_name, topics in self.feature_representations.items():
            if recent_set.issubset(topics) or len(recent_set.intersection(topics)) >= 2:
                coherence_score += 0.3
        
        # Check for progressive difficulty or review patterns
        if len(self.recent_topics) >= 2:
            # If topics repeat (review) or progress logically
            if self.recent_topics[-1] == self.recent_topics[-2]:
                coherence_score += 0.2  # Review pattern
        
        return min(coherence_score, 1.0)
    
    def answer(self, task: Task) -> int:
        """
        Answer a task based on effective skill (accounting for forgetting and transfer).
        
        Returns:
            Index of chosen answer (0-3)
        """
        effective_skill = self._get_effective_skill(task.topic)
        
        # Probability of correct = 0.25 (random) + 0.75 * effective_skill
        prob_correct = 0.25 + 0.75 * effective_skill
        
        if self.rng.random() < prob_correct:
            return task.answer
        else:
            wrong_answers = [i for i in range(4) if i != task.answer]
            return self.rng.choice(wrong_answers)
    
    def learn(self, task: Task) -> bool:
        """
        Learn from a task with PPO-like features.
        
        Features:
        - Transfer learning: Related topics boost learning
        - Exponential learning: Coherent curriculum accelerates learning
        - Multi-step penalty: Harder tasks need more practice
        - Adaptive learning: Learning rate adjusts based on context
        
        Returns:
            Whether answer was correct
        """
        was_correct = (self.answer(task) == task.answer)
        
        topic = task.topic
        difficulty = task.difficulty
        
        # Initialize if new topic
        if topic not in self.topic_skills:
            self.topic_skills[topic] = 0.0
            self.topic_attempts[topic] = 0
            self.last_practice_time[topic] = self.current_time
        
        current_base_skill = self.topic_skills[topic]
        difficulty_factor = self.difficulty_factors.get(difficulty, 0.5)
        
        # PPO-like: Transfer learning boost
        transfer_boost = self._get_transfer_boost(topic)
        
        # PPO-like: Curriculum coherence (exponential learning when guided)
        coherence = self._get_curriculum_coherence()
        curriculum_multiplier = 1.0 + (coherence * 0.5)  # Up to 1.5x with coherent curriculum
        
        # Update recent topics for coherence tracking
        self.recent_topics.append(topic)
        if len(self.recent_topics) > self.recent_topics_window:
            self.recent_topics.pop(0)
        
        # Learning multiplier based on correctness
        if was_correct:
            learning_multiplier = 1.0
        else:
            learning_multiplier = 0.3
        
        # Multi-step penalty for very hard tasks
        steps = self._get_steps_for_difficulty(difficulty)
        step_penalty = 1.0 - (self.multi_step_penalty.get(difficulty, 0.0) * steps)
        
        # Exponential learning when guided, linear when random
        if coherence > 0.6:  # Teacher-guided (coherent)
            # Exponential: faster learning as skills accumulate
            skill_gap = 1.0 - current_base_skill
            exponential_factor = 1.0 + (current_base_skill * 0.5)  # Accelerates with skill
        else:  # Random/progressive (incoherent)
            # Linear: constant learning rate
            skill_gap = 1.0 - current_base_skill
            exponential_factor = 1.0  # No acceleration
        
        skill_increase = (
            self.learning_rate * 
            difficulty_factor * 
            learning_multiplier * 
            skill_gap *
            (1.0 + transfer_boost) *  # Transfer learning
            curriculum_multiplier *  # Curriculum coherence
            step_penalty *  # Multi-step penalty
            exponential_factor  # Exponential vs linear
        )
        
        self.topic_skills[topic] = min(1.0, current_base_skill + skill_increase)
        self.topic_attempts[topic] = self.topic_attempts.get(topic, 0) + 1
        self.last_practice_time[topic] = self.current_time
        self.total_timesteps += 1
        
        return was_correct
    
    def _get_steps_for_difficulty(self, difficulty: str) -> int:
        """Determine number of reasoning steps for a difficulty level."""
        step_map = {
            'trivial': 1,
            'easy': 1,
            'medium': 2,
            'hard': 3,
            'expert': 4,
            'master': 5,
            'grandmaster': 6
        }
        return step_map.get(difficulty, 1)
    
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
        topic_accuracies = {}
        for topic in self.topic_skills.keys():
            effective_skill = self._get_effective_skill(topic)
            topic_accuracies[topic] = 0.25 + 0.75 * effective_skill
        
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
