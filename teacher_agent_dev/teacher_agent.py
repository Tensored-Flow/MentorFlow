"""Teacher Agent using Upper Confidence Bound (UCB) bandit algorithm."""

import numpy as np
from typing import Dict, List
from interfaces import TeacherAction, StudentState, TeacherAgentInterface


def compute_reward(
    accuracy_before: float, 
    accuracy_after: float, 
    difficulty: str, 
    is_review: bool
) -> float:
    """
    Compute reward for teacher action.
    
    Reward structure:
    - Base: improvement in accuracy
    - Bonus: harder tasks encourage pushing boundaries
    - Bonus: successful reviews (spaced repetition)
    - Penalty: wasted reviews (student still remembers perfectly)
    """
    improvement = accuracy_after - accuracy_before
    
    # Bonus for harder tasks (encourage pushing boundaries)
    difficulty_bonus = {'easy': 0.5, 'medium': 1.0, 'hard': 2.0}[difficulty]
    
    # Bonus for successful reviews (spaced repetition)
    review_bonus = 1.0 if (is_review and improvement > 0) else 0.0
    
    # Penalty for wasted reviews (student still remembers perfectly)
    review_penalty = -0.5 if (is_review and accuracy_after > 0.9) else 0.0
    
    return improvement + difficulty_bonus + review_bonus + review_penalty


class TeacherAgent(TeacherAgentInterface):
    """
    Teacher Agent using UCB (Upper Confidence Bound) bandit algorithm.
    
    Action space: 5 topics × 3 difficulties × 2 (new vs review) = 30 actions
    
    UCB formula:
    UCB(a) = estimated_reward(a) + exploration_bonus × sqrt(log(total_pulls) / pulls(a))
    
    Balances exploration (trying new actions) vs exploitation (using known-good actions).
    """
    
    def __init__(self, exploration_bonus: float = 2.0):
        """
        Initialize teacher agent.
        
        Args:
            exploration_bonus: Controls exploration vs exploitation balance.
                              Higher = more exploration (try new actions)
                              Lower = more exploitation (use known-good actions)
        """
        self.exploration_bonus = exploration_bonus
        
        # Define action space
        self.topics = ['history', 'science', 'literature', 'geography', 'current_events']
        self.difficulties = ['easy', 'medium', 'hard']
        self.review_options = [False, True]  # False = new, True = review
        
        # Create all action combinations
        self.actions = [
            (topic, diff, review)
            for topic in self.topics
            for diff in self.difficulties
            for review in self.review_options
        ]
        self.num_actions = len(self.actions)  # Should be 30
        
        # Track statistics per action
        self.action_counts = np.zeros(self.num_actions, dtype=np.float64)
        self.action_rewards = np.zeros(self.num_actions, dtype=np.float64)
        self.total_pulls = 0
    
    def select_action(self, student_state: StudentState) -> TeacherAction:
        """
        Select next action using UCB algorithm.
        
        For each action:
        - If never tried: select it (cold start)
        - Otherwise: compute UCB score and select highest
        """
        # Cold start: try each action at least once
        untried_actions = [i for i in range(self.num_actions) if self.action_counts[i] == 0]
        if untried_actions:
            action_idx = self.total_pulls % len(untried_actions)
            selected_idx = untried_actions[action_idx]
        else:
            # All actions tried - use UCB
            ucb_scores = self._compute_ucb_scores()
            selected_idx = np.argmax(ucb_scores)
        
        return self._index_to_action(selected_idx)
    
    def _compute_ucb_scores(self) -> np.ndarray:
        """Compute UCB score for each action."""
        scores = np.zeros(self.num_actions)
        
        for i in range(self.num_actions):
            if self.action_counts[i] == 0:
                # Never tried - give high score for exploration
                scores[i] = float('inf')
            else:
                # Estimated reward (average so far)
                estimated_reward = self.action_rewards[i] / self.action_counts[i]
                
                # Exploration bonus: sqrt(log(total_pulls) / pulls(action))
                exploration_term = np.sqrt(
                    np.log(max(1, self.total_pulls)) / self.action_counts[i]
                )
                
                # UCB score = estimated reward + exploration bonus
                scores[i] = estimated_reward + self.exploration_bonus * exploration_term
        
        return scores
    
    def update(self, action: TeacherAction, reward: float):
        """
        Update teacher policy based on reward.
        
        Uses running average: new_avg = old_avg + (reward - old_avg) / count
        """
        action_idx = self._action_to_index(action)
        
        # Update statistics
        self.action_counts[action_idx] += 1
        n = self.action_counts[action_idx]
        
        # Running average update
        old_avg = self.action_rewards[action_idx] / max(1, n - 1) if n > 1 else 0.0
        self.action_rewards[action_idx] = (old_avg * (n - 1)) + reward
        
        self.total_pulls += 1
    
    def _action_to_index(self, action: TeacherAction) -> int:
        """Convert TeacherAction to integer index."""
        try:
            topic_idx = self.topics.index(action.topic)
            diff_idx = self.difficulties.index(action.difficulty)
            review_idx = int(action.is_review)
            
            # Encode: topic * (diffs * reviews) + diff * reviews + review
            index = (
                topic_idx * (len(self.difficulties) * len(self.review_options)) +
                diff_idx * len(self.review_options) +
                review_idx
            )
            return index
        except (ValueError, AttributeError):
            raise ValueError(f"Invalid action: {action}")
    
    def _index_to_action(self, index: int) -> TeacherAction:
        """Convert integer index to TeacherAction."""
        if not (0 <= index < self.num_actions):
            raise ValueError(f"Invalid action index: {index}")
        
        # Decode: index -> (topic, difficulty, review)
        review_idx = index % len(self.review_options)
        diff_idx = (index // len(self.review_options)) % len(self.difficulties)
        topic_idx = index // (len(self.difficulties) * len(self.review_options))
        
        return TeacherAction(
            topic=self.topics[topic_idx],
            difficulty=self.difficulties[diff_idx],
            is_review=bool(review_idx)
        )
    
    def get_statistics(self) -> Dict:
        """Get teacher statistics for visualization."""
        return {
            'action_counts': self.action_counts.copy(),
            'action_rewards': self.action_rewards.copy(),
            'actions': self.actions.copy(),
            'topics': self.topics.copy(),
            'difficulties': self.difficulties.copy(),
            'review_options': self.review_options.copy(),
            'total_pulls': self.total_pulls
        }

