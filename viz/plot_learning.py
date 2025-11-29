"""
Learning Curve Visualization for TeachRL
Plot student accuracy and teacher rewards over time.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed. Install with: pip install matplotlib")

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from tasks.task_generator import arm_to_name, FAMILY_NAMES, DIFFICULTY_NAMES


def plot_learning_curve(
    accuracies: List[float],
    title: str = "Student Learning Curve",
    xlabel: str = "Meta-Step",
    ylabel: str = "Accuracy",
    save_path: Optional[str] = None,
    show: bool = True,
    figsize: Tuple[int, int] = (10, 6)
):
    """
    Plot a single learning curve.
    
    Args:
        accuracies: List of accuracy values over time
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        save_path: Optional path to save figure
        show: Whether to display the plot
        figsize: Figure size
    """
    if not HAS_MATPLOTLIB:
        print("Cannot plot: matplotlib not installed")
        return
    
    fig, ax = plt.subplots(figsize=figsize)
    
    steps = list(range(len(accuracies)))
    ax.plot(steps, accuracies, linewidth=2, color='#2196F3')
    
    # Add smoothed line
    if len(accuracies) > 10:
        window = min(20, len(accuracies) // 5)
        smoothed = np.convolve(accuracies, np.ones(window)/window, mode='valid')
        smooth_steps = list(range(window//2, len(accuracies) - window//2))
        ax.plot(smooth_steps, smoothed, '--', linewidth=2, color='#F44336', 
                label=f'Smoothed (window={window})')
        ax.legend()
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    # Add horizontal lines for reference
    ax.axhline(y=0.25, color='gray', linestyle=':', alpha=0.5, label='Random (25%)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    if show:
        plt.show()
    
    plt.close()


def plot_comparison(
    results: Dict[str, Dict],
    metric: str = "student_accuracies",
    title: str = "Strategy Comparison",
    save_path: Optional[str] = None,
    show: bool = True,
    figsize: Tuple[int, int] = (12, 7)
):
    """
    Plot learning curves for multiple strategies.
    
    Args:
        results: Dict mapping strategy name to result dict
        metric: Key in result dict containing the values to plot
        title: Plot title
        save_path: Optional path to save figure
        show: Whether to display
        figsize: Figure size
    """
    if not HAS_MATPLOTLIB:
        print("Cannot plot: matplotlib not installed")
        return
    
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#F44336', '#9C27B0', '#00BCD4']
    
    for i, (strategy, data) in enumerate(results.items()):
        values = data.get(metric, [])
        if not values:
            continue
        
        steps = list(range(len(values)))
        color = colors[i % len(colors)]
        
        # Plot with slight transparency for raw data
        ax.plot(steps, values, alpha=0.3, color=color, linewidth=1)
        
        # Smoothed line
        if len(values) > 10:
            window = min(20, len(values) // 5)
            smoothed = np.convolve(values, np.ones(window)/window, mode='valid')
            smooth_steps = list(range(window//2, len(values) - window//2))
            ax.plot(smooth_steps, smoothed, linewidth=2.5, color=color, label=strategy)
        else:
            ax.plot(steps, values, linewidth=2.5, color=color, label=strategy)
    
    ax.set_xlabel("Meta-Step", fontsize=12)
    ax.set_ylabel("Accuracy" if "accuracy" in metric.lower() else "Value", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    if "accuracy" in metric.lower():
        ax.set_ylim(0, 1)
        ax.axhline(y=0.25, color='gray', linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    if show:
        plt.show()
    
    plt.close()


def plot_teacher_rewards(
    rewards: List[float],
    title: str = "Teacher Reward Over Time",
    save_path: Optional[str] = None,
    show: bool = True,
    figsize: Tuple[int, int] = (10, 6)
):
    """
    Plot teacher rewards with cumulative sum.
    """
    if not HAS_MATPLOTLIB:
        print("Cannot plot: matplotlib not installed")
        return
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[2, 1])
    
    steps = list(range(len(rewards)))
    
    # Per-step rewards
    colors = ['#4CAF50' if r >= 0 else '#F44336' for r in rewards]
    ax1.bar(steps, rewards, color=colors, alpha=0.7, width=1.0)
    ax1.axhline(y=0, color='black', linewidth=0.5)
    ax1.set_ylabel("Reward (Δ Accuracy)", fontsize=11)
    ax1.set_title(title, fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Smoothed rewards
    if len(rewards) > 10:
        window = min(20, len(rewards) // 5)
        smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
        smooth_steps = list(range(window//2, len(rewards) - window//2))
        ax1.plot(smooth_steps, smoothed, color='#2196F3', linewidth=2, label='Rolling avg')
        ax1.legend(loc='upper right')
    
    # Cumulative rewards
    cumulative = np.cumsum(rewards)
    ax2.fill_between(steps, 0, cumulative, alpha=0.3, color='#2196F3')
    ax2.plot(steps, cumulative, color='#2196F3', linewidth=2)
    ax2.set_xlabel("Meta-Step", fontsize=11)
    ax2.set_ylabel("Cumulative", fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    if show:
        plt.show()
    
    plt.close()


def plot_curriculum_timeline(
    selected_arms: List[int],
    title: str = "Curriculum Timeline",
    save_path: Optional[str] = None,
    show: bool = True,
    figsize: Tuple[int, int] = (14, 6)
):
    """
    Plot task selection over time as a timeline.
    """
    if not HAS_MATPLOTLIB:
        print("Cannot plot: matplotlib not installed")
        return
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[3, 1])
    
    steps = list(range(len(selected_arms)))
    
    # Color by difficulty
    difficulty_colors = ['#4CAF50', '#FF9800', '#F44336']  # easy, medium, hard
    colors = [difficulty_colors[arm % 3] for arm in selected_arms]
    
    ax1.scatter(steps, selected_arms, c=colors, s=20, alpha=0.7)
    ax1.set_ylabel("Arm ID", fontsize=11)
    ax1.set_title(title, fontsize=14, fontweight='bold')
    ax1.set_yticks(range(15))
    ax1.set_yticklabels([arm_to_name(i) for i in range(15)], fontsize=8)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Legend for difficulty
    patches = [
        mpatches.Patch(color='#4CAF50', label='Easy'),
        mpatches.Patch(color='#FF9800', label='Medium'),
        mpatches.Patch(color='#F44336', label='Hard'),
    ]
    ax1.legend(handles=patches, loc='upper right')
    
    # Difficulty progression
    difficulties = [arm % 3 for arm in selected_arms]
    window = min(20, len(difficulties) // 5) if len(difficulties) > 10 else 1
    if window > 1:
        smoothed_diff = np.convolve(difficulties, np.ones(window)/window, mode='valid')
        smooth_steps = list(range(window//2, len(difficulties) - window//2))
        ax2.plot(smooth_steps, smoothed_diff, color='#2196F3', linewidth=2)
    else:
        ax2.plot(steps, difficulties, color='#2196F3', linewidth=2)
    
    ax2.set_xlabel("Meta-Step", fontsize=11)
    ax2.set_ylabel("Avg Difficulty", fontsize=11)
    ax2.set_ylim(-0.1, 2.1)
    ax2.set_yticks([0, 1, 2])
    ax2.set_yticklabels(['Easy', 'Medium', 'Hard'])
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    if show:
        plt.show()
    
    plt.close()


def plot_final_accuracy_bar(
    results: Dict[str, Dict],
    title: str = "Final Accuracy by Strategy",
    save_path: Optional[str] = None,
    show: bool = True,
    figsize: Tuple[int, int] = (10, 6)
):
    """
    Bar chart comparing final accuracies.
    """
    if not HAS_MATPLOTLIB:
        print("Cannot plot: matplotlib not installed")
        return
    
    strategies = list(results.keys())
    accuracies = [results[s].get("final_accuracy", 0) for s in strategies]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#F44336', '#9C27B0']
    bars = ax.bar(strategies, accuracies, color=colors[:len(strategies)], alpha=0.8)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.1%}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.axhline(y=0.25, color='gray', linestyle='--', alpha=0.5, label='Random baseline')
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    if show:
        plt.show()
    
    plt.close()


def create_dashboard(
    results: Dict[str, Dict],
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Create a comprehensive dashboard with multiple plots.
    """
    if not HAS_MATPLOTLIB:
        print("Cannot plot: matplotlib not installed")
        return
    
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Learning curves comparison (top left)
    ax1 = fig.add_subplot(2, 2, 1)
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#F44336', '#9C27B0']
    for i, (strategy, data) in enumerate(results.items()):
        accs = data.get("student_accuracies", [])
        if accs:
            ax1.plot(accs, color=colors[i % len(colors)], label=strategy, alpha=0.7)
    ax1.set_xlabel("Meta-Step")
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Learning Curves", fontweight='bold')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    
    # 2. Final accuracy bar chart (top right)
    ax2 = fig.add_subplot(2, 2, 2)
    strategies = list(results.keys())
    final_accs = [results[s].get("final_accuracy", 0) for s in strategies]
    bars = ax2.bar(strategies, final_accs, color=colors[:len(strategies)], alpha=0.8)
    for bar, acc in zip(bars, final_accs):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.0%}', ha='center', va='bottom', fontsize=10)
    ax2.set_ylabel("Final Accuracy")
    ax2.set_title("Final Performance", fontweight='bold')
    ax2.set_ylim(0, 1)
    ax2.axhline(y=0.25, color='gray', linestyle='--', alpha=0.5)
    
    # 3. Cumulative rewards (bottom left)
    ax3 = fig.add_subplot(2, 2, 3)
    for i, (strategy, data) in enumerate(results.items()):
        rewards = data.get("teacher_rewards", [])
        if rewards:
            cumulative = np.cumsum(rewards)
            ax3.plot(cumulative, color=colors[i % len(colors)], label=strategy)
    ax3.set_xlabel("Meta-Step")
    ax3.set_ylabel("Cumulative Reward")
    ax3.set_title("Teacher Cumulative Rewards", fontweight='bold')
    ax3.legend(loc='lower right')
    ax3.grid(True, alpha=0.3)
    
    # 4. Total rewards bar (bottom right)
    ax4 = fig.add_subplot(2, 2, 4)
    total_rewards = [results[s].get("total_teacher_reward", 0) for s in strategies]
    ax4.bar(strategies, total_rewards, color=colors[:len(strategies)], alpha=0.8)
    ax4.set_ylabel("Total Teacher Reward")
    ax4.set_title("Total Rewards", fontweight='bold')
    ax4.axhline(y=0, color='black', linewidth=0.5)
    
    plt.suptitle("TeachRL Experiment Dashboard", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    if show:
        plt.show()
    
    plt.close()


# =============================================================================
# STANDALONE TEST
# =============================================================================

if __name__ == "__main__":
    if not HAS_MATPLOTLIB:
        print("Cannot run visualization test: matplotlib not installed")
        exit(1)
    
    print("=" * 60)
    print("TeachRL Learning Curve Visualization Test")
    print("=" * 60)
    
    # Generate mock data
    np.random.seed(42)
    n_steps = 100
    
    mock_results = {
        "ucb1": {
            "student_accuracies": [0.25 + 0.004 * i + 0.02 * np.random.random() for i in range(n_steps)],
            "teacher_rewards": [0.004 + 0.01 * np.random.randn() for _ in range(n_steps)],
            "selected_arms": list(np.random.randint(0, 15, n_steps)),
            "final_accuracy": 0.68,
            "total_teacher_reward": 0.43,
        },
        "random": {
            "student_accuracies": [0.25 + 0.003 * i + 0.03 * np.random.random() for i in range(n_steps)],
            "teacher_rewards": [0.003 + 0.015 * np.random.randn() for _ in range(n_steps)],
            "selected_arms": list(np.random.randint(0, 15, n_steps)),
            "final_accuracy": 0.55,
            "total_teacher_reward": 0.30,
        },
        "fixed": {
            "student_accuracies": [0.25 + 0.0035 * i + 0.025 * np.random.random() for i in range(n_steps)],
            "teacher_rewards": [0.0035 + 0.012 * np.random.randn() for _ in range(n_steps)],
            "selected_arms": [(i % 5) * 3 + min(i // 33, 2) for i in range(n_steps)],
            "final_accuracy": 0.60,
            "total_teacher_reward": 0.35,
        },
    }
    
    # Test individual plots
    print("\nGenerating plots (will display if matplotlib backend supports it)...")
    
    plot_learning_curve(mock_results["ucb1"]["student_accuracies"], 
                       title="UCB1 Learning Curve", show=False)
    print("✓ Learning curve plot")
    
    plot_comparison(mock_results, title="Strategy Comparison", show=False)
    print("✓ Comparison plot")
    
    plot_teacher_rewards(mock_results["ucb1"]["teacher_rewards"], show=False)
    print("✓ Teacher rewards plot")
    
    plot_curriculum_timeline(mock_results["fixed"]["selected_arms"], show=False)
    print("✓ Curriculum timeline plot")
    
    plot_final_accuracy_bar(mock_results, show=False)
    print("✓ Final accuracy bar chart")
    
    create_dashboard(mock_results, show=False)
    print("✓ Dashboard")
    
    print("\n✓ All visualization tests complete!")
