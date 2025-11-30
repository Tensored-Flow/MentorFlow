"""
Beautiful, publication-quality visualizations for student learning.

Creates comprehensive plots showing learning curves, retention, and efficiency.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List
from student_metrics import StudentMetrics

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150


def plot_learning_curve(
    metrics: StudentMetrics,
    save_path: str = 'student_learning_curve.png'
):
    """Plot overall accuracy over time with smoothing."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    iterations = metrics.iterations
    accuracies = metrics.overall_accuracies
    
    # Plot raw accuracy
    ax.plot(iterations, accuracies, alpha=0.3, color='blue', label='Raw accuracy')
    
    # Plot smoothed (moving average)
    window = 20
    if len(accuracies) >= window:
        smoothed = np.convolve(accuracies, np.ones(window)/window, mode='valid')
        ax.plot(iterations[window-1:], smoothed, linewidth=2, color='blue', label=f'Smoothed ({window}-step MA)')
    
    # Add milestone lines
    ax.axhline(y=0.5, color='green', linestyle='--', alpha=0.5, label='50% accuracy')
    ax.axhline(y=0.7, color='orange', linestyle='--', alpha=0.5, label='70% accuracy')
    ax.axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='80% mastery')
    
    ax.set_xlabel('Training Iteration', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Student Learning Curve', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 Saved learning curve to {save_path}")


def plot_per_topic_learning(
    metrics: StudentMetrics,
    save_path: str = 'topic_learning_curves.png'
):
    """Plot learning curves for each topic separately."""
    topics = list(metrics.per_topic_accuracies.keys())
    
    if not topics:
        print("⚠️ No topic data to plot")
        return
    
    n_topics = len(topics)
    n_cols = 3
    n_rows = (n_topics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    axes = axes.flatten() if n_topics > 1 else [axes]
    
    for i, topic in enumerate(topics):
        ax = axes[i]
        accs = metrics.per_topic_accuracies[topic]
        
        ax.plot(accs, linewidth=2, color=f'C{i}')
        ax.axhline(y=0.7, color='red', linestyle='--', alpha=0.5)
        ax.set_title(f'{topic.capitalize()}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Practice Sessions')
        ax.set_ylabel('Accuracy')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)
    
    # Hide extra subplots
    for i in range(n_topics, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle('Per-Topic Learning Curves', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 Saved per-topic curves to {save_path}")


def plot_retention_analysis(
    metrics: StudentMetrics,
    save_path: str = 'retention_analysis.png'
):
    """Plot retention factors over time for each topic."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for topic, retentions in metrics.retention_factors.items():
        if retentions:
            ax.plot(retentions, label=topic, linewidth=2, alpha=0.7)
    
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='50% retention threshold')
    ax.set_xlabel('Training Iteration', fontsize=12)
    ax.set_ylabel('Retention Factor', fontsize=12)
    ax.set_title('Memory Retention Analysis (Forgetting Curves)', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 Saved retention analysis to {save_path}")


def plot_difficulty_progression(
    metrics: StudentMetrics,
    save_path: str = 'difficulty_progression.png'
):
    """Visualize how task difficulty changes over time."""
    diff_map = {'easy': 1, 'medium': 2, 'hard': 3}
    diff_values = [diff_map.get(d, 2) for d in metrics.difficulties_seen]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.scatter(range(len(diff_values)), diff_values, alpha=0.5, s=20)
    
    window = 20
    if len(diff_values) >= window:
        smoothed = np.convolve(diff_values, np.ones(window)/window, mode='valid')
        ax.plot(range(window-1, len(diff_values)), smoothed, 
               color='red', linewidth=2, label=f'Moving average ({window}-step)')
    
    ax.set_yticks([1, 2, 3])
    ax.set_yticklabels(['Easy', 'Medium', 'Hard'])
    ax.set_xlabel('Training Iteration', fontsize=12)
    ax.set_ylabel('Task Difficulty', fontsize=12)
    ax.set_title('Task Difficulty Progression', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 Saved difficulty progression to {save_path}")


def plot_topic_distribution(
    metrics: StudentMetrics,
    save_path: str = 'topic_distribution.png'
):
    """Show distribution of topics practiced."""
    from collections import Counter
    
    topic_counts = Counter(metrics.topics_seen)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    topics = list(topic_counts.keys())
    counts = list(topic_counts.values())
    
    ax1.bar(topics, counts, color='steelblue', edgecolor='black', alpha=0.7)
    ax1.set_xlabel('Topic', fontsize=12)
    ax1.set_ylabel('Number of Tasks', fontsize=12)
    ax1.set_title('Topic Practice Distribution', fontsize=14, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3, axis='y')
    
    ax2.pie(counts, labels=topics, autopct='%1.1f%%', startangle=90)
    ax2.set_title('Topic Practice Proportion', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 Saved topic distribution to {save_path}")


def plot_sample_efficiency(
    metrics: StudentMetrics,
    save_path: str = 'sample_efficiency.png'
):
    """Show how many tasks needed to reach accuracy milestones."""
    milestones = [0.5, 0.6, 0.7, 0.8]
    tasks_needed = []
    
    for milestone in milestones:
        tasks = metrics.compute_sample_efficiency(milestone)
        tasks_needed.append(tasks if tasks < len(metrics.iterations) else None)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    reached_milestones = [(m, t) for m, t in zip(milestones, tasks_needed) if t is not None]
    
    if reached_milestones:
        milestones_reached, tasks = zip(*reached_milestones)
        
        ax.bar(range(len(milestones_reached)), tasks, color='coral', edgecolor='black', alpha=0.7)
        ax.set_xticks(range(len(milestones_reached)))
        ax.set_xticklabels([f'{m*100:.0f}%' for m in milestones_reached])
        ax.set_xlabel('Accuracy Milestone', fontsize=12)
        ax.set_ylabel('Tasks Required', fontsize=12)
        ax.set_title('Sample Efficiency: Tasks to Reach Milestones', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        for i, t in enumerate(tasks):
            ax.text(i, t + 5, str(t), ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 Saved sample efficiency to {save_path}")


def create_comprehensive_report(
    metrics: StudentMetrics,
    output_dir: str = 'student_visualizations'
):
    """Generate all visualizations and save to directory."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n📊 Generating comprehensive student report in {output_dir}/\n")
    
    plot_learning_curve(metrics, f'{output_dir}/learning_curve.png')
    plot_per_topic_learning(metrics, f'{output_dir}/topic_curves.png')
    plot_retention_analysis(metrics, f'{output_dir}/retention.png')
    plot_difficulty_progression(metrics, f'{output_dir}/difficulty.png')
    plot_topic_distribution(metrics, f'{output_dir}/topics.png')
    plot_sample_efficiency(metrics, f'{output_dir}/efficiency.png')
    
    # Print summary
    summary = metrics.get_summary_statistics()
    print("\n" + "="*60)
    print("STUDENT LEARNING SUMMARY")
    print("="*60)
    print(f"Total Tasks:          {summary['total_tasks']}")
    print(f"Final Accuracy:       {summary['final_accuracy']:.3f}")
    print(f"Max Accuracy:         {summary['max_accuracy']:.3f}")
    print(f"Mean Accuracy:        {summary['mean_accuracy']:.3f}")
    print(f"Learning Rate:        {summary['learning_rate']:.4f}")
    print(f"Tasks to 70%:         {summary['sample_efficiency_70']}")
    print(f"Tasks to 80%:         {summary['sample_efficiency_80']}")
    print(f"Topics Practiced:     {summary['topics_practiced']}")
    print("="*60)
    
    print(f"\n✅ Report complete! Check {output_dir}/ for all visualizations.")

