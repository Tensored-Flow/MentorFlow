"""
Teacher Bandit for TeachRL
Multi-armed bandit that selects which task type the student should train on.
90 arms = 18 families × 5 difficulties
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tasks.task_generator import NUM_FAMILIES, NUM_DIFFICULTIES, arm_to_name


NUM_ARMS = NUM_FAMILIES * NUM_DIFFICULTIES


class BanditStrategy(Enum):
    """Available bandit strategies."""
    UCB1 = "ucb1"
    THOMPSON_SAMPLING = "thompson_sampling"
    EPSILON_GREEDY = "epsilon_greedy"
    RANDOM = "random"


@dataclass
class ArmStats:
    """Statistics for a single arm."""
    pulls: int = 0
    total_reward: float = 0.0
    rewards: List[float] = field(default_factory=list)
    
    @property
    def mean_reward(self) -> float:
        if self.pulls == 0:
            return 0.0
        return self.total_reward / self.pulls
    
    def update(self, reward: float):
        self.pulls += 1
        self.total_reward += reward
        self.rewards.append(reward)


class TeacherBandit:
    """
    Multi-armed bandit teacher that learns to select optimal task types.
    
    The teacher's goal is to select tasks that maximize student learning
    (measured as improvement in evaluation accuracy).
    """
    
    def __init__(
        self,
        strategy: BanditStrategy = BanditStrategy.UCB1,
        exploration_param: float = 2.0,
        epsilon: float = 0.1,
        seed: Optional[int] = None
    ):
        """
        Initialize teacher bandit.
        
        Args:
            strategy: Bandit algorithm to use
            exploration_param: UCB exploration parameter (c)
            epsilon: Epsilon for epsilon-greedy
            seed: Random seed
        """
        self.strategy = strategy
        self.exploration_param = exploration_param
        self.epsilon = epsilon
        self._rng = np.random.default_rng(seed)
        
        # Arm statistics
        self.arms: List[ArmStats] = [ArmStats() for _ in range(NUM_ARMS)]
        self.total_pulls = 0
        
        # History for visualization
        self.selection_history: List[int] = []
        self.reward_history: List[float] = []
    
    def select_arm(self) -> int:
        """
        Select which arm (task type) to pull next.
        
        Returns:
            arm_id (0-14)
        """
        if self.strategy == BanditStrategy.RANDOM:
            arm = self._select_random()
        elif self.strategy == BanditStrategy.EPSILON_GREEDY:
            arm = self._select_epsilon_greedy()
        elif self.strategy == BanditStrategy.UCB1:
            arm = self._select_ucb1()
        elif self.strategy == BanditStrategy.THOMPSON_SAMPLING:
            arm = self._select_thompson()
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        
        self.selection_history.append(arm)
        return arm
    
    def update(self, arm_id: int, reward: float):
        """
        Update arm statistics after observing reward.
        
        Args:
            arm_id: The arm that was pulled
            reward: Observed reward (student improvement)
        """
        self.arms[arm_id].update(reward)
        self.total_pulls += 1
        self.reward_history.append(reward)
    
    def _select_random(self) -> int:
        """Random arm selection."""
        return int(self._rng.integers(0, NUM_ARMS))
    
    def _select_epsilon_greedy(self) -> int:
        """Epsilon-greedy selection."""
        if self._rng.random() < self.epsilon:
            return self._select_random()
        
        # Greedy: select arm with highest mean reward
        # Handle ties randomly, prioritize unexplored
        mean_rewards = [arm.mean_reward for arm in self.arms]
        unpulled = [i for i, arm in enumerate(self.arms) if arm.pulls == 0]
        
        if unpulled:
            return int(self._rng.choice(unpulled))
        
        max_reward = max(mean_rewards)
        best_arms = [i for i, r in enumerate(mean_rewards) if r == max_reward]
        return int(self._rng.choice(best_arms))
    
    def _select_ucb1(self) -> int:
        """
        UCB1 selection.
        UCB = mean_reward + c * sqrt(ln(t) / n_i)
        """
        # First, try each arm at least once
        for i, arm in enumerate(self.arms):
            if arm.pulls == 0:
                return i
        
        ucb_values = []
        log_total = np.log(self.total_pulls)
        
        for arm in self.arms:
            exploitation = arm.mean_reward
            exploration = self.exploration_param * np.sqrt(log_total / arm.pulls)
            ucb_values.append(exploitation + exploration)
        
        # Break ties randomly
        max_ucb = max(ucb_values)
        best_arms = [i for i, v in enumerate(ucb_values) if v == max_ucb]
        return int(self._rng.choice(best_arms))
    
    def _select_thompson(self) -> int:
        """
        Thompson Sampling with Beta distribution.
        Treats rewards as binary (positive improvement = success).
        """
        samples = []
        
        for arm in self.arms:
            if arm.pulls == 0:
                # Uniform prior: Beta(1, 1)
                sample = self._rng.beta(1, 1)
            else:
                # Count positive rewards as successes
                successes = sum(1 for r in arm.rewards if r > 0)
                failures = arm.pulls - successes
                # Beta posterior: Beta(1 + successes, 1 + failures)
                sample = self._rng.beta(1 + successes, 1 + failures)
            samples.append(sample)
        
        return int(np.argmax(samples))
    
    def get_arm_stats(self) -> Dict[str, Dict]:
        """Get statistics for all arms."""
        stats = {}
        for i, arm in enumerate(self.arms):
            stats[arm_to_name(i)] = {
                "pulls": arm.pulls,
                "mean_reward": arm.mean_reward,
                "total_reward": arm.total_reward,
            }
        return stats
    
    def get_curriculum_heatmap(self) -> np.ndarray:
        """
        Get pull counts as 2D array for heatmap visualization.
        Shape: (NUM_FAMILIES, NUM_DIFFICULTIES)
        """
        heatmap = np.zeros((NUM_FAMILIES, NUM_DIFFICULTIES))
        for arm_id, arm in enumerate(self.arms):
            family = arm_id // NUM_DIFFICULTIES
            difficulty = arm_id % NUM_DIFFICULTIES
            heatmap[family, difficulty] = arm.pulls
        return heatmap
    
    def get_reward_heatmap(self) -> np.ndarray:
        """
        Get mean rewards as 2D array for heatmap.
        Shape: (NUM_FAMILIES, NUM_DIFFICULTIES)
        """
        heatmap = np.zeros((NUM_FAMILIES, NUM_DIFFICULTIES))
        for arm_id, arm in enumerate(self.arms):
            family = arm_id // NUM_DIFFICULTIES
            difficulty = arm_id % NUM_DIFFICULTIES
            heatmap[family, difficulty] = arm.mean_reward
        return heatmap
    
    def reset(self):
        """Reset all statistics."""
        self.arms = [ArmStats() for _ in range(NUM_ARMS)]
        self.total_pulls = 0
        self.selection_history = []
        self.reward_history = []


class FixedCurriculumTeacher:
    """
    Baseline teacher that follows a fixed curriculum.
    Progresses through difficulties: easy → medium → hard.
    """
    
    def __init__(
        self,
        steps_per_difficulty: int = 50,
        cycle_families: bool = True,
        seed: Optional[int] = None
    ):
        """
        Args:
            steps_per_difficulty: Steps before advancing difficulty
            cycle_families: If True, cycles through families at each difficulty
            seed: Random seed
        """
        self.steps_per_difficulty = steps_per_difficulty
        self.cycle_families = cycle_families
        self._rng = np.random.default_rng(seed)
        
        self.step_count = 0
        self.selection_history: List[int] = []
        self.reward_history: List[float] = []
    
    def select_arm(self) -> int:
        """Select arm based on fixed curriculum."""
        # Determine current difficulty based on step count
        difficulty = min(
            self.step_count // self.steps_per_difficulty,
            NUM_DIFFICULTIES - 1
        )
        
        # Select family
        if self.cycle_families:
            family = self.step_count % NUM_FAMILIES
        else:
            family = int(self._rng.integers(0, NUM_FAMILIES))
        
        arm_id = family * NUM_DIFFICULTIES + difficulty
        self.selection_history.append(arm_id)
        self.step_count += 1
        
        return arm_id
    
    def update(self, arm_id: int, reward: float):
        """Record reward (no learning)."""
        self.reward_history.append(reward)
    
    def reset(self):
        """Reset curriculum."""
        self.step_count = 0
        self.selection_history = []
        self.reward_history = []


class RandomTeacher:
    """Baseline teacher that selects tasks uniformly at random."""
    
    def __init__(self, seed: Optional[int] = None):
        self._rng = np.random.default_rng(seed)
        self.selection_history: List[int] = []
        self.reward_history: List[float] = []
    
    def select_arm(self) -> int:
        arm = int(self._rng.integers(0, NUM_ARMS))
        self.selection_history.append(arm)
        return arm
    
    def update(self, arm_id: int, reward: float):
        self.reward_history.append(reward)
    
    def reset(self):
        self.selection_history = []
        self.reward_history = []


# =============================================================================
# STANDALONE TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("TeachRL Teacher Bandit Test")
    print("=" * 60)
    
    # Simulate rewards for testing
    def simulate_reward(arm_id: int, rng: np.random.Generator) -> float:
        """Simulate student improvement (some arms are 'better')."""
        # Make some arms have higher expected reward
        family = arm_id // NUM_DIFFICULTIES
        difficulty = arm_id % NUM_DIFFICULTIES
        
        # Easier tasks have higher improvement for beginners
        base_prob = 0.3 - 0.1 * difficulty
        if rng.random() < base_prob:
            return rng.uniform(0.01, 0.05)
        return rng.uniform(-0.02, 0.02)
    
    rng = np.random.default_rng(42)
    
    # Test UCB1
    print("\n--- UCB1 Bandit (100 steps) ---")
    teacher = TeacherBandit(strategy=BanditStrategy.UCB1, seed=42)
    
    for _ in range(100):
        arm = teacher.select_arm()
        reward = simulate_reward(arm, rng)
        teacher.update(arm, reward)
    
    print("Arm statistics:")
    for name, stats in teacher.get_arm_stats().items():
        if stats["pulls"] > 0:
            print(f"  {name}: pulls={stats['pulls']}, mean_reward={stats['mean_reward']:.4f}")
    
    print(f"\nCurriculum heatmap (pulls):")
    heatmap = teacher.get_curriculum_heatmap()
    print(heatmap)
    
    # Test Thompson Sampling
    print("\n--- Thompson Sampling (100 steps) ---")
    teacher2 = TeacherBandit(strategy=BanditStrategy.THOMPSON_SAMPLING, seed=42)
    
    for _ in range(100):
        arm = teacher2.select_arm()
        reward = simulate_reward(arm, rng)
        teacher2.update(arm, reward)
    
    print(f"Total reward: {sum(teacher2.reward_history):.4f}")
    print(f"Most pulled arm: {arm_to_name(np.argmax([a.pulls for a in teacher2.arms]))}")
    
    # Test Fixed Curriculum
    print("\n--- Fixed Curriculum (30 steps) ---")
    fixed_teacher = FixedCurriculumTeacher(steps_per_difficulty=10, seed=42)
    
    for _ in range(30):
        arm = fixed_teacher.select_arm()
        print(f"Step {fixed_teacher.step_count}: {arm_to_name(arm)}")
    
    # Test Random
    print("\n--- Random Teacher (20 steps) ---")
    random_teacher = RandomTeacher(seed=42)
    selections = [random_teacher.select_arm() for _ in range(20)]
    print(f"Selections: {[arm_to_name(a) for a in selections[:5]]}...")
    
    print("\n✓ All teacher tests passed!")
