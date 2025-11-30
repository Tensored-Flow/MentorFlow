"""Visualization utilities for Teacher Agent system."""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List
from teacher_agent import TeacherAgent


def plot_learning_curves(history: Dict, save_path: str = 'learning_curves.png'):
    """
    Plot student accuracy and teacher reward over time.
    
    Args:
        history: Dictionary with 'iterations', 'student_accuracies', 'teacher_rewards'
        save_path: Where to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    iterations = history['iterations']
    
    # Plot student accuracy
    ax1.plot(iterations, history['student_accuracies'], label='Student Accuracy', linewidth=2)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Student Learning Curve')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_ylim([0, 1])
    
    # Plot teacher reward (smoothed)
    rewards = np.array(history['teacher_rewards'])
    window = 50
    if len(rewards) > window:
        smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
        smoothed_iterations = iterations[window-1:]
        ax2.plot(smoothed_iterations, smoothed, label=f'Smoothed Reward (window={window})', linewidth=2)
        ax2.plot(iterations, rewards, alpha=0.3, label='Raw Reward', linewidth=0.5)
    else:
        ax2.plot(iterations, rewards, label='Reward', linewidth=2)
    
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Reward')
    ax2.set_title('Teacher Reward Over Time')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved learning curves to {save_path}")
    plt.close()


def plot_curriculum_heatmap(history: Dict, save_path: str = 'curriculum_heatmap.png'):
    """
    Visualize teacher's curriculum choices over time.
    
    Args:
        history: Dictionary with 'iterations', 'topics', 'difficulties', 'is_reviews'
        save_path: Where to save the plot
    """
    topics = list(set(history['topics']))
    topics.sort()
    
    # Create grid: time (iterations) vs topics
    num_iterations = len(history['iterations'])
    num_topics = len(topics)
    
    # Map difficulty to numeric value
    difficulty_map = {'easy': 1, 'medium': 2, 'hard': 3}
    
    # Create heatmap data
    heatmap_data = np.zeros((num_topics, num_iterations))
    
    for i, (topic, difficulty, is_review) in enumerate(zip(
        history['topics'], 
        history['difficulties'], 
        history['is_reviews']
    )):
        topic_idx = topics.index(topic)
        diff_value = difficulty_map[difficulty]
        if is_review:
            diff_value = 0.5  # Mark reviews differently
        heatmap_data[topic_idx, i] = diff_value
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    im = ax.imshow(heatmap_data, aspect='auto', cmap='viridis', interpolation='nearest')
    
    ax.set_yticks(range(num_topics))
    ax.set_yticklabels(topics)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Topic')
    ax.set_title('Curriculum Heatmap (Light=Easy/Review, Dark=Hard)')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Difficulty (0.5=Review, 1=Easy, 2=Medium, 3=Hard)')
    
    # Sample iterations for x-axis labels
    if num_iterations > 20:
        step = num_iterations // 10
        ax.set_xticks(range(0, num_iterations, step))
        ax.set_xticklabels(range(0, num_iterations, step))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved curriculum heatmap to {save_path}")
    plt.close()


def plot_action_distributions(teacher: TeacherAgent, save_path: str = 'action_dist.png'):
    """
    Show which actions teacher prefers.
    
    Args:
        teacher: Trained TeacherAgent
        save_path: Where to save the plot
    """
    stats = teacher.get_statistics()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Topic distribution
    topic_counts = {}
    for idx, count in enumerate(stats['action_counts']):
        if count > 0:
            action = teacher._index_to_action(idx)
            topic_counts[action.topic] = topic_counts.get(action.topic, 0) + count
    
    ax = axes[0, 0]
    topics = list(topic_counts.keys())
    counts = list(topic_counts.values())
    ax.bar(topics, counts)
    ax.set_xlabel('Topic')
    ax.set_ylabel('Count')
    ax.set_title('Topic Selection Distribution')
    ax.tick_params(axis='x', rotation=45)
    
    # 2. Difficulty distribution
    difficulty_counts = {'easy': 0, 'medium': 0, 'hard': 0}
    for idx, count in enumerate(stats['action_counts']):
        if count > 0:
            action = teacher._index_to_action(idx)
            difficulty_counts[action.difficulty] += count
    
    ax = axes[0, 1]
    difficulties = list(difficulty_counts.keys())
    counts = list(difficulty_counts.values())
    ax.bar(difficulties, counts)
    ax.set_xlabel('Difficulty')
    ax.set_ylabel('Count')
    ax.set_title('Difficulty Selection Distribution')
    
    # 3. Review vs New
    review_counts = {'New': 0, 'Review': 0}
    for idx, count in enumerate(stats['action_counts']):
        if count > 0:
            action = teacher._index_to_action(idx)
            key = 'Review' if action.is_review else 'New'
            review_counts[key] += count
    
    ax = axes[1, 0]
    labels = list(review_counts.keys())
    sizes = list(review_counts.values())
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    ax.set_title('New vs Review Distribution')
    
    # 4. Average reward per topic
    topic_rewards = {}
    for idx in range(len(stats['action_counts'])):
        if stats['action_counts'][idx] > 0:
            action = teacher._index_to_action(idx)
            avg_reward = stats['action_rewards'][idx] / stats['action_counts'][idx]
            topic_rewards[action.topic] = topic_rewards.get(action.topic, []) + [avg_reward]
    
    # Compute mean reward per topic
    topic_avg_rewards = {topic: np.mean(rewards) for topic, rewards in topic_rewards.items()}
    
    ax = axes[1, 1]
    topics = list(topic_avg_rewards.keys())
    rewards = list(topic_avg_rewards.values())
    ax.bar(topics, rewards)
    ax.set_xlabel('Topic')
    ax.set_ylabel('Average Reward')
    ax.set_title('Average Reward per Topic')
    ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved action distributions to {save_path}")
    plt.close()


def plot_comparison(histories: Dict[str, Dict], save_path: str = 'comparison.png'):
    """
    Compare teacher vs baselines.
    
    Args:
        histories: Dictionary mapping strategy name to history dict
                   e.g., {'teacher': history1, 'random': history2, 'fixed': history3}
        save_path: Where to save the plot
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot accuracy comparison
    ax = axes[0]
    for name, history in histories.items():
        iterations = history['iterations']
        accuracies = history['student_accuracies']
        ax.plot(iterations, accuracies, label=name, linewidth=2)
    
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Accuracy')
    ax.set_title('Student Accuracy Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    # Plot reward comparison (smoothed)
    ax = axes[1]
    window = 50
    for name, history in histories.items():
        rewards = np.array(history['teacher_rewards'])
        iterations = history['iterations']
        
        if len(rewards) > window:
            smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
            smoothed_iterations = iterations[window-1:]
            ax.plot(smoothed_iterations, smoothed, label=f'{name} (smoothed)', linewidth=2)
        else:
            ax.plot(iterations, rewards, label=name, linewidth=2)
    
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Reward')
    ax.set_title('Teacher Reward Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved comparison plot to {save_path}")
    plt.close()


if __name__ == "__main__":
    # Example usage
    print("This module provides visualization functions.")
    print("Import and use them with training results:")
    print()
    print("  from train_teacher import train_teacher")
    print("  from visualize import *")
    print()
    print("  history, teacher, student = train_teacher(num_iterations=500)")
    print("  plot_learning_curves(history)")
    print("  plot_curriculum_heatmap(history)")
    print("  plot_action_distributions(teacher)")

