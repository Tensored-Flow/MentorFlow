"""Upper Confidence Bound (UCB1) teacher for curriculum selection."""

import numpy as np


class TeacherUCB:
    """Classic UCB1 bandit algorithm for selecting among task arms."""

    def __init__(self, num_arms: int):
        if num_arms <= 0:
            raise ValueError("num_arms must be positive.")
        self.num_arms = num_arms
        self.counts = np.zeros(num_arms, dtype=np.int64)
        self.values = np.zeros(num_arms, dtype=np.float64)

    def select_arm(self) -> int:
        """Select the next arm to pull."""
        # Pull each arm at least once.
        for arm in range(self.num_arms):
            if self.counts[arm] == 0:
                return arm

        total_counts = np.sum(self.counts)
        confidence_bounds = np.sqrt(2 * np.log(total_counts) / self.counts)
        ucb_scores = self.values + confidence_bounds
        return int(np.argmax(ucb_scores))

    def update(self, arm: int, reward: float) -> None:
        """Update running estimates after observing a reward."""
        if arm < 0 or arm >= self.num_arms:
            raise IndexError(f"Arm index {arm} out of bounds for {self.num_arms} arms.")

        self.counts[arm] += 1
        n = self.counts[arm]
        value = self.values[arm]
        # Running average update rule.
        self.values[arm] = value + (reward - value) / n
