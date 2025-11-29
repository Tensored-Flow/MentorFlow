"""
Evaluation Metrics for TeachRL
Functions to compute and analyze training results.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from tasks.task_generator import (
    TaskSpec, generate_eval_dataset, arm_to_name,
    NUM_FAMILIES, NUM_DIFFICULTIES, FAMILY_NAMES, DIFFICULTY_NAMES
)


def compute_accuracy(predictions: List[int], correct_actions: List[int]) -> float:
    """Compute accuracy from predictions."""
    if not predictions:
        return 0.0
    correct = sum(p == c for p, c in zip(predictions, correct_actions))
    return correct / len(predictions)


def compute_accuracy_by_type(
    eval_tasks: List[TaskSpec],
    predictions: List[int]
) -> Dict[str, float]:
    """
    Compute accuracy broken down by task type.
    
    Returns:
        Dict mapping task type name to accuracy
    """
    # Group by type
    type_results: Dict[int, List[bool]] = {i: [] for i in range(15)}
    
    for task, pred in zip(eval_tasks, predictions):
        arm_id = task.family_id * NUM_DIFFICULTIES + task.difficulty_id
        correct = (pred == task.correct_action)
        type_results[arm_id].append(correct)
    
    # Compute accuracy per type
    accuracies = {}
    for arm_id, results in type_results.items():
        if results:
            accuracies[arm_to_name(arm_id)] = sum(results) / len(results)
        else:
            accuracies[arm_to_name(arm_id)] = 0.0
    
    return accuracies


def compute_accuracy_by_family(
    eval_tasks: List[TaskSpec],
    predictions: List[int]
) -> Dict[str, float]:
    """Compute accuracy by task family (aggregated across difficulties)."""
    family_results: Dict[int, List[bool]] = {i: [] for i in range(NUM_FAMILIES)}
    
    for task, pred in zip(eval_tasks, predictions):
        correct = (pred == task.correct_action)
        family_results[task.family_id].append(correct)
    
    return {
        FAMILY_NAMES[fid]: sum(r) / len(r) if r else 0.0
        for fid, r in family_results.items()
    }


def compute_accuracy_by_difficulty(
    eval_tasks: List[TaskSpec],
    predictions: List[int]
) -> Dict[str, float]:
    """Compute accuracy by difficulty level (aggregated across families)."""
    diff_results: Dict[int, List[bool]] = {i: [] for i in range(NUM_DIFFICULTIES)}
    
    for task, pred in zip(eval_tasks, predictions):
        correct = (pred == task.correct_action)
        diff_results[task.difficulty_id].append(correct)
    
    return {
        DIFFICULTY_NAMES[did]: sum(r) / len(r) if r else 0.0
        for did, r in diff_results.items()
    }


def compute_learning_speed(
    accuracies: List[float],
    threshold: float = 0.5
) -> Optional[int]:
    """
    Compute number of steps to reach accuracy threshold.
    
    Returns:
        Step number when threshold reached, or None if never reached
    """
    for i, acc in enumerate(accuracies):
        if acc >= threshold:
            return i
    return None


def compute_area_under_curve(accuracies: List[float]) -> float:
    """Compute area under the learning curve (higher = better)."""
    if not accuracies:
        return 0.0
    return np.trapz(accuracies) / len(accuracies)


def compute_curriculum_entropy(selection_history: List[int]) -> float:
    """
    Compute entropy of curriculum (arm selection distribution).
    Higher entropy = more uniform exploration.
    """
    if not selection_history:
        return 0.0
    
    counts = np.zeros(15)
    for arm in selection_history:
        counts[arm] += 1
    
    probs = counts / counts.sum()
    probs = probs[probs > 0]  # Remove zeros for log
    
    return -np.sum(probs * np.log2(probs))


def compute_curriculum_progression(
    selection_history: List[int],
    window_size: int = 20
) -> List[float]:
    """
    Compute average difficulty over time using rolling window.
    
    Returns:
        List of average difficulty values
    """
    if not selection_history:
        return []
    
    difficulties = [arm % NUM_DIFFICULTIES for arm in selection_history]
    progression = []
    
    for i in range(len(difficulties)):
        start = max(0, i - window_size + 1)
        window = difficulties[start:i + 1]
        progression.append(np.mean(window))
    
    return progression


def compute_improvement_rate(
    teacher_rewards: List[float],
    window_size: int = 20
) -> List[float]:
    """
    Compute rolling average of teacher rewards (improvement rate).
    """
    if not teacher_rewards:
        return []
    
    rates = []
    for i in range(len(teacher_rewards)):
        start = max(0, i - window_size + 1)
        window = teacher_rewards[start:i + 1]
        rates.append(np.mean(window))
    
    return rates


def compare_strategies(
    results: Dict[str, Dict],
    metric: str = "final_accuracy"
) -> List[Tuple[str, float]]:
    """
    Compare strategies by a specific metric.
    
    Args:
        results: Dict mapping strategy name to result dict
        metric: Metric to compare by
    
    Returns:
        List of (strategy, value) sorted by value descending
    """
    scores = [(name, r.get(metric, 0)) for name, r in results.items()]
    return sorted(scores, key=lambda x: x[1], reverse=True)


def generate_summary_report(
    results: Dict[str, Dict],
    output_path: Optional[str] = None
) -> str:
    """
    Generate a text summary report of experiment results.
    
    Args:
        results: Dict mapping strategy name to result dict
        output_path: Optional path to save report
    
    Returns:
        Report string
    """
    lines = []
    lines.append("=" * 60)
    lines.append("TeachRL Experiment Summary Report")
    lines.append("=" * 60)
    lines.append("")
    
    # Comparison table
    lines.append("STRATEGY COMPARISON")
    lines.append("-" * 60)
    lines.append(f"{'Strategy':<20} {'Final Acc':>12} {'AUC':>10} {'Reward':>12}")
    lines.append("-" * 60)
    
    for strategy, r in results.items():
        acc = r.get("final_accuracy", 0)
        auc = compute_area_under_curve(r.get("student_accuracies", []))
        reward = r.get("total_teacher_reward", 0)
        lines.append(f"{strategy:<20} {acc:>11.2%} {auc:>10.4f} {reward:>12.4f}")
    
    lines.append("")
    
    # Rankings
    lines.append("RANKINGS")
    lines.append("-" * 60)
    
    rankings = compare_strategies(results, "final_accuracy")
    lines.append("By Final Accuracy:")
    for i, (name, val) in enumerate(rankings, 1):
        lines.append(f"  {i}. {name}: {val:.2%}")
    
    lines.append("")
    
    # Learning speed comparison
    lines.append("LEARNING SPEED (steps to 50% accuracy)")
    lines.append("-" * 60)
    for strategy, r in results.items():
        accs = r.get("student_accuracies", [])
        speed = compute_learning_speed(accs, 0.5)
        speed_str = str(speed) if speed is not None else "Never"
        lines.append(f"  {strategy}: {speed_str}")
    
    lines.append("")
    lines.append("=" * 60)
    
    report = "\n".join(lines)
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(report)
    
    return report


# =============================================================================
# STANDALONE TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("TeachRL Evaluation Metrics Test")
    print("=" * 60)
    
    # Generate test data
    eval_tasks = generate_eval_dataset(tasks_per_type=5, seed=42)
    
    # Simulate predictions (mix of correct and random)
    np.random.seed(42)
    predictions = []
    for task in eval_tasks:
        if np.random.random() < 0.6:  # 60% correct
            predictions.append(task.correct_action)
        else:
            predictions.append(np.random.randint(0, 4))
    
    # Test metrics
    print(f"\nOverall accuracy: {compute_accuracy(predictions, [t.correct_action for t in eval_tasks]):.2%}")
    
    print("\nAccuracy by type:")
    type_acc = compute_accuracy_by_type(eval_tasks, predictions)
    for name, acc in sorted(type_acc.items())[:5]:
        print(f"  {name}: {acc:.0%}")
    print("  ...")
    
    print("\nAccuracy by family:")
    family_acc = compute_accuracy_by_family(eval_tasks, predictions)
    for name, acc in family_acc.items():
        print(f"  {name}: {acc:.0%}")
    
    print("\nAccuracy by difficulty:")
    diff_acc = compute_accuracy_by_difficulty(eval_tasks, predictions)
    for name, acc in diff_acc.items():
        print(f"  {name}: {acc:.0%}")
    
    # Test curriculum metrics
    selection_history = list(np.random.randint(0, 15, 100))
    print(f"\nCurriculum entropy: {compute_curriculum_entropy(selection_history):.2f}")
    
    # Test learning metrics
    accuracies = [0.25 + 0.003 * i + 0.02 * np.random.random() for i in range(100)]
    print(f"Learning speed (to 50%): {compute_learning_speed(accuracies, 0.5)}")
    print(f"Area under curve: {compute_area_under_curve(accuracies):.4f}")
    
    # Test report generation
    mock_results = {
        "ucb1": {"final_accuracy": 0.72, "student_accuracies": accuracies, "total_teacher_reward": 0.47},
        "random": {"final_accuracy": 0.58, "student_accuracies": [a * 0.8 for a in accuracies], "total_teacher_reward": 0.33},
    }
    
    print("\n" + generate_summary_report(mock_results))
    
    print("\n✓ Evaluation metrics test complete!")
