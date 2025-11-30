"""
Comprehensive metrics tracking for student learning.

Tracks overall accuracy, per-topic performance, retention, and efficiency metrics.
"""

from dataclasses import dataclass, field
from typing import List, Dict
import numpy as np
from collections import defaultdict


@dataclass
class StudentMetrics:
    """Comprehensive metrics for student learning."""
    
    # Time series data
    iterations: List[int] = field(default_factory=list)
    overall_accuracies: List[float] = field(default_factory=list)
    per_topic_accuracies: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    
    # Per-iteration details
    tasks_seen: List[str] = field(default_factory=list)  # task_id
    topics_seen: List[str] = field(default_factory=list)
    difficulties_seen: List[str] = field(default_factory=list)
    was_correct: List[bool] = field(default_factory=list)
    
    # Retention tracking
    retention_factors: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    
    # Learning efficiency
    tasks_to_mastery: Dict[str, int] = field(default_factory=dict)  # topic -> num tasks
    
    def log_iteration(
        self,
        iteration: int,
        overall_acc: float,
        topic_accs: Dict[str, float],
        task: 'Task',
        correct: bool,
        retention_factors: Dict[str, float]
    ):
        """Log a single training iteration."""
        self.iterations.append(iteration)
        self.overall_accuracies.append(overall_acc)
        
        for topic, acc in topic_accs.items():
            self.per_topic_accuracies[topic].append(acc)
        
        self.tasks_seen.append(task.task_id)
        self.topics_seen.append(task.topic)
        self.difficulties_seen.append(task.difficulty)
        self.was_correct.append(correct)
        
        for topic, retention in retention_factors.items():
            self.retention_factors[topic].append(retention)
    
    def compute_learning_rate(self, window: int = 50) -> float:
        """Compute average improvement per task (last N tasks)."""
        if len(self.overall_accuracies) < window:
            return 0.0
        
        recent_accs = self.overall_accuracies[-window:]
        improvements = np.diff(recent_accs)
        return np.mean(improvements)
    
    def compute_sample_efficiency(self, target_accuracy: float = 0.7) -> int:
        """Number of tasks needed to reach target accuracy."""
        for i, acc in enumerate(self.overall_accuracies):
            if acc >= target_accuracy:
                return i
        return len(self.overall_accuracies)  # Not reached yet
    
    def compute_topic_mastery_times(self, mastery_threshold: float = 0.8) -> Dict[str, int]:
        """Tasks needed to master each topic."""
        mastery_times = {}
        
        for topic, accs in self.per_topic_accuracies.items():
            for i, acc in enumerate(accs):
                if acc >= mastery_threshold:
                    mastery_times[topic] = i
                    break
        
        return mastery_times
    
    def get_summary_statistics(self) -> Dict:
        """Get overall summary statistics."""
        return {
            'total_tasks': len(self.iterations),
            'final_accuracy': self.overall_accuracies[-1] if self.overall_accuracies else 0.0,
            'max_accuracy': max(self.overall_accuracies) if self.overall_accuracies else 0.0,
            'mean_accuracy': np.mean(self.overall_accuracies) if self.overall_accuracies else 0.0,
            'learning_rate': self.compute_learning_rate(),
            'sample_efficiency_70': self.compute_sample_efficiency(0.7),
            'sample_efficiency_80': self.compute_sample_efficiency(0.8),
            'topics_practiced': len(self.per_topic_accuracies),
            'topic_mastery_times': self.compute_topic_mastery_times()
        }

