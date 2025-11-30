"""Personalized learning profile tracking for MentorFlow."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from typing import Deque, Dict, List, Optional, Tuple
import time


@dataclass
class HistoryEntry:
    timestamp: float
    family: str
    difficulty: str
    correct: bool
    arm_id: Optional[int]

    def to_dict(self) -> Dict[str, object]:
        data = asdict(self)
        return data


class LearningProfile:
    """Tracks personalised learning performance across task families/difficulties."""

    def __init__(self, families: List[str], difficulties: List[str]):
        self.families = families
        self.difficulties = difficulties

        self.family_stats: Dict[str, Dict[str, float]] = {
            family: {"correct": 0.0, "attempts": 0.0} for family in families
        }
        self.difficulty_stats: Dict[str, Dict[str, float]] = {
            diff: {"correct": 0.0, "attempts": 0.0} for diff in difficulties
        }
        self.attempts_per_family: Dict[str, int] = defaultdict(int)
        self.current_streak: int = 0
        self.mastery_scores: Dict[str, float] = defaultdict(float)
        self.history: Deque[HistoryEntry] = deque(maxlen=20)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def update_profile(self, *, family: str, difficulty: str, arm_id: Optional[int], correct: bool) -> None:
        """Update profile stats after a task attempt."""
        # Family stats
        fam = self.family_stats.setdefault(family, {"correct": 0.0, "attempts": 0.0})
        fam["attempts"] += 1
        fam["correct"] += 1 if correct else 0
        self.attempts_per_family[family] += 1

        # Difficulty stats
        diff = self.difficulty_stats.setdefault(difficulty, {"correct": 0.0, "attempts": 0.0})
        diff["attempts"] += 1
        diff["correct"] += 1 if correct else 0

        # Streak
        if correct:
            self.current_streak += 1
        else:
            self.current_streak = 0

        # Rolling mastery (EMA)
        prev = self.mastery_scores.get(family, 0.0)
        alpha = 0.2
        self.mastery_scores[family] = (1 - alpha) * prev + alpha * (1.0 if correct else 0.0)

        # History
        self.history.append(
            HistoryEntry(
                timestamp=time.time(),
                family=family,
                difficulty=difficulty,
                correct=correct,
                arm_id=arm_id,
            )
        )

    def compute_mastery(self) -> Dict[str, float]:
        return {family: round(score, 3) for family, score in self.mastery_scores.items()}

    def detect_weaknesses(self, min_attempts: int = 3) -> Dict[str, float]:
        """Return weakness scores (1 - accuracy) for families with enough attempts."""
        weaknesses: Dict[str, float] = {}
        for family, stats in self.family_stats.items():
            attempts = stats["attempts"]
            if attempts < min_attempts:
                continue
            acc = stats["correct"] / attempts if attempts else 0.0
            weaknesses[family] = round(1 - acc, 3)
        return weaknesses

    def recommend_next_family(self) -> Optional[str]:
        weaknesses = self.detect_weaknesses(min_attempts=2)
        if weaknesses:
            return max(weaknesses, key=lambda k: weaknesses[k])
        # fallback -> least attempts
        if not self.attempts_per_family:
            return self.families[0] if self.families else None
        return min(self.attempts_per_family, key=lambda k: self.attempts_per_family[k])

    def recommend_next_difficulty(self, target_family: Optional[str] = None) -> Optional[str]:
        if target_family is None:
            target_family = self.recommend_next_family()
        if target_family is None:
            return self.difficulties[0] if self.difficulties else None

        # pick difficulty with lowest accuracy within family based on history
        diff_scores: Dict[str, List[bool]] = defaultdict(list)
        for entry in self.history:
            if entry.family == target_family:
                diff_scores[entry.difficulty].append(entry.correct)
        if not diff_scores:
            return self.difficulties[0] if self.difficulties else None
        weakness_by_diff = {
            diff: (1 - sum(vals) / len(vals))
            for diff, vals in diff_scores.items()
            if vals
        }
        if weakness_by_diff:
            return max(weakness_by_diff, key=lambda k: weakness_by_diff[k])
        return self.difficulties[0] if self.difficulties else None

    def serialize(self) -> Dict[str, object]:
        return {
            "accuracy_per_family": self._accuracy_map(self.family_stats),
            "accuracy_per_difficulty": self._accuracy_map(self.difficulty_stats),
            "attempts_per_family": dict(self.attempts_per_family),
            "streak": self.current_streak,
            "mastery_scores": self.compute_mastery(),
            "recent_history": [entry.to_dict() for entry in list(self.history)],
            "weakness_scores": self.detect_weaknesses(min_attempts=1),
            "recommended_family": self.recommend_next_family(),
            "recommended_difficulty": self.recommend_next_difficulty(),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _accuracy_map(stats: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        data: Dict[str, float] = {}
        for key, values in stats.items():
            attempts = values["attempts"]
            acc = values["correct"] / attempts if attempts else 0.0
            data[key] = round(acc, 3)
        return data


__all__ = ["LearningProfile"]
