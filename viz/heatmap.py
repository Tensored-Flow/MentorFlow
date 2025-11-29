"""
Heatmap Visualization for TeachRL
Visualize curriculum selection patterns and performance by task type.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed. Install with: pip install matplotlib")

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from tasks.task_generator import FAMILY_NAMES, DIFFICULTY_NAMES, NUM_FAMILIES, NUM_DIFFICULTIES


def plot_curriculum_heatmap(
    heatmap: np.ndarray,
    title: str = "Curriculum Heatmap (Task Selections)",
    cmap: str = "YlOrRd",
    save_path: Optional[str] = None,
    show: bool = True,
    figsize: Tuple[int, int] = (10, 8),
    annotate: bool = True
):
    """
    Plot heatmap of task selections (families × difficulties).
    
    Args:
        heatmap: 2D array of shape (NUM_FAMILIES, NUM_DIFFICULTIES)
        title: Plot title
        cmap: Colormap name
        save_path: Optional path to save figure
        show: Whether to display
        figsize: Figure size
        annotate: Whether to show values in cells
    """
    if not HAS_MATPLOTLIB:
        print("Cannot plot: matplotlib not installed")
        return
    
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(heatmap, cmap=cmap, aspect='auto')
    
    # Axes
    ax.set_xticks(range(NUM_DIFFICULTIES))
    ax.set_xticklabels(DIFFICULTY_NAMES, fontsize=12)
    ax.set_yticks(range(NUM_FAMILIES))
    ax.set_yticklabels(FAMILY_NAMES, fontsize=12)
    
    ax.set_xlabel("Difficulty", fontsize=13)
    ax.set_ylabel("Task Family", fontsize=13)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Count", fontsize=12)
    
    # Annotate cells
    if annotate:
        for i in range(NUM_FAMILIES):
            for j in range(NUM_DIFFICULTIES):
                value = heatmap[i, j]
                # Choose text color based on background
                text_color = 'white' if value > heatmap.max() * 0.6 else 'black'
                ax.text(j, i, f'{int(value)}', ha='center', va='center',
                       fontsize=14, fontweight='bold', color=text_color)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    if show:
        plt.show()
    
    plt.close()


def plot_accuracy_heatmap(
    accuracy_by_type: Dict[str, float],
    title: str = "Accuracy by Task Type",
    cmap: str = "RdYlGn",
    save_path: Optional[str] = None,
    show: bool = True,
    figsize: Tuple[int, int] = (10, 8)
):
    """
    Plot heatmap of accuracy by task type.
    
    Args:
        accuracy_by_type: Dict mapping task type name to accuracy
        title: Plot title
        cmap: Colormap name
        save_path: Optional path to save figure
        show: Whether to display
        figsize: Figure size
    """
    if not HAS_MATPLOTLIB:
        print("Cannot plot: matplotlib not installed")
        return
    
    # Convert dict to 2D array
    heatmap = np.zeros((NUM_FAMILIES, NUM_DIFFICULTIES))
    for name, acc in accuracy_by_type.items():
        parts = name.rsplit('_', 1)
        if len(parts) == 2:
            family_name, diff_name = parts
            if family_name in FAMILY_NAMES and diff_name in DIFFICULTY_NAMES:
                fam_idx = FAMILY_NAMES.index(family_name)
                diff_idx = DIFFICULTY_NAMES.index(diff_name)
                heatmap[fam_idx, diff_idx] = acc
    
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(heatmap, cmap=cmap, aspect='auto', vmin=0, vmax=1)
    
    ax.set_xticks(range(NUM_DIFFICULTIES))
    ax.set_xticklabels(DIFFICULTY_NAMES, fontsize=12)
    ax.set_yticks(range(NUM_FAMILIES))
    ax.set_yticklabels(FAMILY_NAMES, fontsize=12)
    
    ax.set_xlabel("Difficulty", fontsize=13)
    ax.set_ylabel("Task Family", fontsize=13)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Accuracy", fontsize=12)
    
    # Annotate
    for i in range(NUM_FAMILIES):
        for j in range(NUM_DIFFICULTIES):
            value = heatmap[i, j]
            text_color = 'white' if value < 0.5 else 'black'
            ax.text(j, i, f'{value:.0%}', ha='center', va='center',
                   fontsize=12, fontweight='bold', color=text_color)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    if show:
        plt.show()
    
    plt.close()


def plot_reward_heatmap(
    reward_heatmap: np.ndarray,
    title: str = "Mean Teacher Reward by Task Type",
    save_path: Optional[str] = None,
    show: bool = True,
    figsize: Tuple[int, int] = (10, 8)
):
    """
    Plot heatmap of mean teacher rewards.
    """
    if not HAS_MATPLOTLIB:
        print("Cannot plot: matplotlib not installed")
        return
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Use diverging colormap centered at 0
    vmax = max(abs(reward_heatmap.min()), abs(reward_heatmap.max()))
    vmax = max(vmax, 0.01)  # Ensure non-zero range
    
    im = ax.imshow(reward_heatmap, cmap='RdBu', aspect='auto', 
                   vmin=-vmax, vmax=vmax)
    
    ax.set_xticks(range(NUM_DIFFICULTIES))
    ax.set_xticklabels(DIFFICULTY_NAMES, fontsize=12)
    ax.set_yticks(range(NUM_FAMILIES))
    ax.set_yticklabels(FAMILY_NAMES, fontsize=12)
    
    ax.set_xlabel("Difficulty", fontsize=13)
    ax.set_ylabel("Task Family", fontsize=13)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Mean Reward", fontsize=12)
    
    # Annotate
    for i in range(NUM_FAMILIES):
        for j in range(NUM_DIFFICULTIES):
            value = reward_heatmap[i, j]
            text_color = 'white' if abs(value) > vmax * 0.6 else 'black'
            ax.text(j, i, f'{value:.3f}', ha='center', va='center',
                   fontsize=11, fontweight='bold', color=text_color)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    if show:
        plt.show()
    
    plt.close()


def plot_comparison_heatmaps(
    results: Dict[str, Dict],
    metric: str = "curriculum_heatmap",
    title: str = "Curriculum Comparison",
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Plot heatmaps for multiple strategies side by side.
    """
    if not HAS_MATPLOTLIB:
        print("Cannot plot: matplotlib not installed")
        return
    
    n_strategies = len(results)
    fig, axes = plt.subplots(1, n_strategies, figsize=(5 * n_strategies, 6))
    
    if n_strategies == 1:
        axes = [axes]
    
    cmap = "YlOrRd" if "curriculum" in metric else "RdYlGn"
    
    # Find global max for consistent color scale
    all_heatmaps = []
    for data in results.values():
        hm = data.get(metric)
        if hm is not None:
            if isinstance(hm, list):
                hm = np.array(hm)
            all_heatmaps.append(hm)
    
    if not all_heatmaps:
        print(f"No {metric} data found in results")
        return
    
    vmax = max(h.max() for h in all_heatmaps)
    vmin = 0 if "curriculum" in metric else min(h.min() for h in all_heatmaps)
    
    for ax, (strategy, data) in zip(axes, results.items()):
        hm = data.get(metric)
        if hm is None:
            continue
        if isinstance(hm, list):
            hm = np.array(hm)
        
        im = ax.imshow(hm, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
        
        ax.set_xticks(range(NUM_DIFFICULTIES))
        ax.set_xticklabels(['E', 'M', 'H'], fontsize=10)
        ax.set_yticks(range(NUM_FAMILIES))
        ax.set_yticklabels([f[:4] for f in FAMILY_NAMES], fontsize=9)
        ax.set_title(strategy, fontsize=12, fontweight='bold')
        
        # Annotate
        for i in range(NUM_FAMILIES):
            for j in range(NUM_DIFFICULTIES):
                val = hm[i, j]
                if "curriculum" in metric:
                    text = f'{int(val)}'
                else:
                    text = f'{val:.0%}'
                text_color = 'white' if val > vmax * 0.6 else 'black'
                ax.text(j, i, text, ha='center', va='center',
                       fontsize=9, color=text_color)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    if show:
        plt.show()
    
    plt.close()


def plot_selection_history_heatmap(
    selected_arms: List[int],
    window_size: int = 20,
    title: str = "Task Selection Over Time",
    save_path: Optional[str] = None,
    show: bool = True,
    figsize: Tuple[int, int] = (14, 6)
):
    """
    Plot selection frequency as a time-windowed heatmap.
    """
    if not HAS_MATPLOTLIB:
        print("Cannot plot: matplotlib not installed")
        return
    
    n_steps = len(selected_arms)
    n_windows = n_steps // window_size
    
    # Build time-windowed counts
    time_heatmap = np.zeros((15, n_windows))
    for w in range(n_windows):
        start = w * window_size
        end = start + window_size
        for arm in selected_arms[start:end]:
            time_heatmap[arm, w] += 1
    
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(time_heatmap, cmap='YlOrRd', aspect='auto')
    
    ax.set_xlabel(f"Time Window (size={window_size})", fontsize=12)
    ax.set_ylabel("Task Type", fontsize=12)
    ax.set_yticks(range(15))
    ax.set_yticklabels([f"{FAMILY_NAMES[i//3][:4]}_{DIFFICULTY_NAMES[i%3][0]}" 
                       for i in range(15)], fontsize=8)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.colorbar(im, ax=ax, shrink=0.8, label='Count')
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
    print("TeachRL Heatmap Visualization Test")
    print("=" * 60)
    
    # Generate mock data
    np.random.seed(42)
    
    # Curriculum heatmap (selection counts)
    curriculum_heatmap = np.random.randint(5, 25, (NUM_FAMILIES, NUM_DIFFICULTIES))
    
    # Accuracy by type
    accuracy_by_type = {}
    for fam in FAMILY_NAMES:
        for diff in DIFFICULTY_NAMES:
            # Harder tasks have lower accuracy
            base_acc = 0.7 - 0.15 * DIFFICULTY_NAMES.index(diff)
            accuracy_by_type[f"{fam}_{diff}"] = base_acc + 0.1 * np.random.random()
    
    # Reward heatmap
    reward_heatmap = np.random.randn(NUM_FAMILIES, NUM_DIFFICULTIES) * 0.02
    
    # Mock comparison results
    mock_results = {
        "ucb1": {
            "curriculum_heatmap": curriculum_heatmap,
            "accuracy_by_type": accuracy_by_type,
        },
        "random": {
            "curriculum_heatmap": np.random.randint(10, 15, (NUM_FAMILIES, NUM_DIFFICULTIES)),
            "accuracy_by_type": {k: v * 0.8 for k, v in accuracy_by_type.items()},
        },
        "fixed": {
            "curriculum_heatmap": np.array([[20, 0, 0], [20, 0, 0], [0, 20, 0], 
                                            [0, 20, 0], [0, 0, 20]]),
            "accuracy_by_type": {k: v * 0.9 for k, v in accuracy_by_type.items()},
        },
    }
    
    # Test plots
    print("\nGenerating heatmaps...")
    
    plot_curriculum_heatmap(curriculum_heatmap, show=False)
    print("✓ Curriculum heatmap")
    
    plot_accuracy_heatmap(accuracy_by_type, show=False)
    print("✓ Accuracy heatmap")
    
    plot_reward_heatmap(reward_heatmap, show=False)
    print("✓ Reward heatmap")
    
    plot_comparison_heatmaps(mock_results, metric="curriculum_heatmap", show=False)
    print("✓ Comparison heatmaps")
    
    # Selection history
    selected_arms = list(np.random.randint(0, 15, 200))
    plot_selection_history_heatmap(selected_arms, show=False)
    print("✓ Selection history heatmap")
    
    print("\n✓ All heatmap tests complete!")
