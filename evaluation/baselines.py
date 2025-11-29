"""
Baseline Evaluation for TeachRL
Implementations and utilities for baseline comparisons.
"""

import numpy as np
from typing import Dict, List, Optional
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from tasks.task_generator import (
    TaskSpec, generate_eval_dataset, arm_to_name,
    NUM_FAMILIES, NUM_DIFFICULTIES
)


class RandomBaseline:
    """Baseline that predicts randomly."""
    
    def __init__(self, seed: Optional[int] = None):
        self._rng = np.random.default_rng(seed)
    
    def predict(self, obs: np.ndarray) -> int:
        return int(self._rng.integers(0, 4))
    
    def predict_batch(self, obs_batch: np.ndarray) -> np.ndarray:
        return self._rng.integers(0, 4, size=len(obs_batch))
    
    def evaluate(self, eval_tasks: List[TaskSpec]) -> float:
        correct = 0
        for task in eval_tasks:
            if self.predict(None) == task.correct_action:
                correct += 1
        return correct / len(eval_tasks) if eval_tasks else 0.0


class MajorityBaseline:
    """Baseline that always predicts the most common answer."""
    
    def __init__(self, default_action: int = 0):
        self.default_action = default_action
    
    def fit(self, tasks: List[TaskSpec]):
        """Fit to training data by finding most common answer."""
        if not tasks:
            return
        
        counts = [0, 0, 0, 0]
        for task in tasks:
            counts[task.correct_action] += 1
        
        self.default_action = int(np.argmax(counts))
    
    def predict(self, obs: np.ndarray) -> int:
        return self.default_action
    
    def evaluate(self, eval_tasks: List[TaskSpec]) -> float:
        correct = sum(1 for t in eval_tasks if t.correct_action == self.default_action)
        return correct / len(eval_tasks) if eval_tasks else 0.0


class FamilySpecificBaseline:
    """Baseline that learns the most common answer per task family."""
    
    def __init__(self):
        self.family_actions: Dict[int, int] = {}
    
    def fit(self, tasks: List[TaskSpec]):
        """Fit by finding most common answer per family."""
        family_counts: Dict[int, List[int]] = {i: [0, 0, 0, 0] for i in range(NUM_FAMILIES)}
        
        for task in tasks:
            family_counts[task.family_id][task.correct_action] += 1
        
        for fam_id, counts in family_counts.items():
            self.family_actions[fam_id] = int(np.argmax(counts))
    
    def predict(self, obs: np.ndarray) -> int:
        # Extract family from one-hot encoding (first 5 elements)
        family_id = int(np.argmax(obs[:5]))
        return self.family_actions.get(family_id, 0)
    
    def evaluate(self, eval_tasks: List[TaskSpec]) -> float:
        correct = 0
        for task in eval_tasks:
            obs = np.array(task.obs_vec)
            if self.predict(obs) == task.correct_action:
                correct += 1
        return correct / len(eval_tasks) if eval_tasks else 0.0


def evaluate_all_baselines(
    eval_tasks: List[TaskSpec],
    train_tasks: Optional[List[TaskSpec]] = None,
    seed: int = 42
) -> Dict[str, float]:
    """
    Evaluate all baselines on the evaluation set.
    
    Args:
        eval_tasks: Tasks to evaluate on
        train_tasks: Optional training tasks for fitted baselines
        seed: Random seed
    
    Returns:
        Dict mapping baseline name to accuracy
    """
    if train_tasks is None:
        train_tasks = eval_tasks
    
    results = {}
    
    # Random baseline
    random_baseline = RandomBaseline(seed=seed)
    # Average over multiple runs for stability
    random_accs = [random_baseline.evaluate(eval_tasks) for _ in range(10)]
    results["random"] = np.mean(random_accs)
    
    # Majority baseline
    majority = MajorityBaseline()
    majority.fit(train_tasks)
    results["majority"] = majority.evaluate(eval_tasks)
    
    # Family-specific baseline
    family_specific = FamilySpecificBaseline()
    family_specific.fit(train_tasks)
    results["family_specific"] = family_specific.evaluate(eval_tasks)
    
    # Theoretical random baseline
    results["theoretical_random"] = 0.25  # 1/4 choices
    
    return results


def compute_baseline_comparison(
    model_accuracy: float,
    baseline_accuracies: Dict[str, float]
) -> Dict[str, float]:
    """
    Compute improvement over baselines.
    
    Returns:
        Dict mapping baseline name to relative improvement
    """
    improvements = {}
    for name, baseline_acc in baseline_accuracies.items():
        if baseline_acc > 0:
            improvement = (model_accuracy - baseline_acc) / baseline_acc
        else:
            improvement = float('inf') if model_accuracy > 0 else 0.0
        improvements[name] = improvement
    
    return improvements


def print_baseline_report(
    model_accuracy: float,
    baseline_accuracies: Dict[str, float]
):
    """Print a formatted baseline comparison report."""
    print("\n" + "=" * 50)
    print("BASELINE COMPARISON")
    print("=" * 50)
    print(f"{'Method':<25} {'Accuracy':>12} {'vs Model':>12}")
    print("-" * 50)
    
    print(f"{'MODEL (trained)':^25} {model_accuracy:>11.2%} {'---':>12}")
    print("-" * 50)
    
    improvements = compute_baseline_comparison(model_accuracy, baseline_accuracies)
    
    for name, acc in sorted(baseline_accuracies.items(), key=lambda x: x[1], reverse=True):
        diff = model_accuracy - acc
        print(f"{name:<25} {acc:>11.2%} {diff:>+11.2%}")
    
    print("=" * 50)
    
    # Summary
    best_baseline = max(baseline_accuracies.items(), key=lambda x: x[1])
    gap = model_accuracy - best_baseline[1]
    print(f"\nModel beats best baseline ({best_baseline[0]}) by {gap:+.2%}")


# =============================================================================
# STANDALONE TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("TeachRL Baselines Test")
    print("=" * 60)
    
    # Generate evaluation data
    eval_tasks = generate_eval_dataset(tasks_per_type=10, seed=42)
    print(f"Eval dataset size: {len(eval_tasks)}")
    
    # Evaluate all baselines
    print("\nEvaluating baselines...")
    baseline_results = evaluate_all_baselines(eval_tasks, seed=42)
    
    print("\nBaseline accuracies:")
    for name, acc in sorted(baseline_results.items(), key=lambda x: x[1], reverse=True):
        print(f"  {name}: {acc:.2%}")
    
    # Simulate a trained model accuracy
    mock_model_accuracy = 0.65
    
    print_baseline_report(mock_model_accuracy, baseline_results)
    
    # Test individual baselines
    print("\n--- Individual Baseline Tests ---")
    
    # Random
    random_b = RandomBaseline(seed=123)
    print(f"Random single prediction: {random_b.predict(None)}")
    
    # Majority
    majority_b = MajorityBaseline()
    majority_b.fit(eval_tasks)
    print(f"Majority class action: {majority_b.default_action}")
    
    # Family-specific
    family_b = FamilySpecificBaseline()
    family_b.fit(eval_tasks)
    print(f"Family-specific actions: {family_b.family_actions}")
    
    print("\n✓ Baselines test complete!")
