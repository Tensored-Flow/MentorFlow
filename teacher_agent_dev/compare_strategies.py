"""
Compare three training strategies:
1. Random: Random questions until student can pass difficult questions
2. Progressive: Easy → Medium → Hard within each family sequentially
3. Teacher: RL teacher agent learns optimal curriculum
"""

import numpy as np
from typing import Dict, Tuple
from interfaces import Task

from mock_student import MockStudentAgent
from mock_task_generator import MockTaskGenerator
from teacher_agent import TeacherAgent, compute_reward
from train_teacher import train_teacher


def evaluate_difficult_questions(student: MockStudentAgent, generator: MockTaskGenerator, num_questions: int = 20) -> float:
    """
    Evaluate student on difficult questions from all topics.
    
    Returns:
        Accuracy on difficult questions (0.0 to 1.0)
    """
    topics = generator.get_available_topics()
    eval_tasks = []
    
    # Generate difficult questions from all topics
    questions_per_topic = max(1, num_questions // len(topics))
    for topic in topics:
        for _ in range(questions_per_topic):
            eval_tasks.append(generator.generate_task(topic, 'hard'))
    
    return student.evaluate(eval_tasks)


def train_strategy_random(num_iterations: int = 500, seed: int = 42, target_accuracy: float = 0.75) -> Dict:
    """
    Strategy 1: Random questions until student can confidently pass difficult questions.
    
    Args:
        num_iterations: Maximum iterations to train
        seed: Random seed
        target_accuracy: Target accuracy on difficult questions to consider "passing"
    
    Returns:
        Training history dictionary
    """
    import random
    rng = random.Random(seed)
    
    student = MockStudentAgent(learning_rate=0.15, forgetting_rate=0.05, seed=seed)
    generator = MockTaskGenerator(seed=seed)
    
    topics = generator.get_available_topics()
    difficulties = generator.get_available_difficulties()
    
    # Evaluation on difficult questions
    hard_eval_tasks = []
    for topic in topics:
        for _ in range(5):  # 5 hard questions per topic
            hard_eval_tasks.append(generator.generate_task(topic, 'hard'))
    
    history = {
        'iterations': [],
        'student_accuracies': [],
        'difficult_accuracies': [],  # Accuracy on hard questions
        'teacher_rewards': [],
        'topics': [],
        'difficulties': [],
        'strategy': 'random'
    }
    
    for iteration in range(num_iterations):
        # Random action
        topic = rng.choice(topics)
        difficulty = rng.choice(difficulties)
        
        task = generator.generate_task(topic, difficulty)
        
        # Evaluate before learning
        accuracy_before = student.evaluate(hard_eval_tasks)
        
        # Student learns
        student.learn(task)
        
        # Evaluate after learning
        accuracy_after = student.evaluate(hard_eval_tasks)
        general_accuracy = student.evaluate([
            generator.generate_task(t, 'medium')
            for t in topics
            for _ in range(2)
        ])
        
        student.advance_time(1.0)
        
        # Track metrics
        history['iterations'].append(iteration)
        history['student_accuracies'].append(general_accuracy)
        history['difficult_accuracies'].append(accuracy_after)
        history['teacher_rewards'].append(accuracy_after - accuracy_before)
        history['topics'].append(topic)
        history['difficulties'].append(difficulty)
        
        # Check if we've reached target (optional early stopping)
        if accuracy_after >= target_accuracy and iteration > 50:  # Give at least 50 iterations
            if 'reached_target' not in locals():
                print(f"  Random strategy reached target accuracy {target_accuracy:.2f} at iteration {iteration}")
                reached_target = True
    
    return history


def train_strategy_progressive(num_iterations: int = 500, seed: int = 42) -> Dict:
    """
    Strategy 2: Progressive difficulty within each family.
    Easy → Medium → Hard for each topic, then move to next topic.
    
    Args:
        num_iterations: Number of iterations
        seed: Random seed
    
    Returns:
        Training history dictionary
    """
    student = MockStudentAgent(learning_rate=0.15, forgetting_rate=0.05, seed=seed)
    generator = MockTaskGenerator(seed=seed)
    
    topics = generator.get_available_topics()
    difficulties = ['easy', 'medium', 'hard']
    
    # Evaluation on difficult questions
    hard_eval_tasks = []
    for topic in topics:
        for _ in range(5):
            hard_eval_tasks.append(generator.generate_task(topic, 'hard'))
    
    history = {
        'iterations': [],
        'student_accuracies': [],
        'difficult_accuracies': [],
        'teacher_rewards': [],
        'topics': [],
        'difficulties': [],
        'strategy': 'progressive'
    }
    
    # Progressive curriculum: cycle through topics, increase difficulty over time
    # Structure: For each topic, do easy → medium → hard
    questions_per_difficulty = num_iterations // (len(topics) * len(difficulties))
    
    for iteration in range(num_iterations):
        # Determine current phase
        phase = iteration // questions_per_difficulty
        topic_idx = (phase // len(difficulties)) % len(topics)
        diff_idx = phase % len(difficulties)
        
        topic = topics[topic_idx]
        difficulty = difficulties[diff_idx]
        
        task = generator.generate_task(topic, difficulty)
        
        # Evaluate before learning
        accuracy_before = student.evaluate(hard_eval_tasks)
        
        # Student learns
        student.learn(task)
        
        # Evaluate after learning
        accuracy_after = student.evaluate(hard_eval_tasks)
        general_accuracy = student.evaluate([
            generator.generate_task(t, 'medium')
            for t in topics
            for _ in range(2)
        ])
        
        student.advance_time(1.0)
        
        # Track metrics
        history['iterations'].append(iteration)
        history['student_accuracies'].append(general_accuracy)
        history['difficult_accuracies'].append(accuracy_after)
        history['teacher_rewards'].append(accuracy_after - accuracy_before)
        history['topics'].append(topic)
        history['difficulties'].append(difficulty)
    
    return history


def train_strategy_teacher(num_iterations: int = 500, seed: int = 42) -> Dict:
    """
    Strategy 3: RL Teacher Agent learns optimal curriculum.
    
    Args:
        num_iterations: Number of iterations
        seed: Random seed
    
    Returns:
        Training history dictionary with difficult_accuracies added
    """
    # Initialize components
    teacher = TeacherAgent(exploration_bonus=2.0)
    student = MockStudentAgent(learning_rate=0.15, forgetting_rate=0.05, seed=seed)
    generator = MockTaskGenerator(seed=seed)
    
    topics = generator.get_available_topics()
    
    # Create evaluation sets
    eval_tasks = [
        generator.generate_task(topic, 'medium')
        for topic in topics
        for _ in range(3)
    ]
    
    # Create difficult question evaluation set
    hard_eval_tasks = [
        generator.generate_task(topic, 'hard')
        for topic in topics
        for _ in range(5)
    ]
    
    # Track metrics
    history = {
        'iterations': [],
        'student_accuracies': [],
        'difficult_accuracies': [],
        'teacher_rewards': [],
        'actions': [],
        'topics': [],
        'difficulties': [],
        'is_reviews': [],
        'strategy': 'teacher'
    }
    
    for iteration in range(num_iterations):
        # 1. Get student state
        student_state = student.get_state()
        
        # 2. Teacher selects action
        action = teacher.select_action(student_state)
        
        # 3. Generate task
        if action.is_review:
            task = generator.generate_task(action.topic, 'medium')
        else:
            task = generator.generate_task(action.topic, action.difficulty)
        
        # 4. Evaluate student BEFORE learning
        accuracy_before = student.evaluate(eval_tasks)
        difficult_acc_before = student.evaluate(hard_eval_tasks)
        
        # 5. Student learns from task
        student.learn(task)
        
        # 6. Evaluate student AFTER learning
        accuracy_after = student.evaluate(eval_tasks)
        difficult_acc_after = student.evaluate(hard_eval_tasks)
        
        # 7. Compute reward for teacher
        reward = compute_reward(
            accuracy_before, 
            accuracy_after, 
            action.difficulty, 
            action.is_review
        )
        
        # 8. Update teacher's policy
        teacher.update(action, reward)
        
        # 9. Time passes (for forgetting)
        student.advance_time(1.0)
        
        # 10. Log metrics
        history['iterations'].append(iteration)
        history['student_accuracies'].append(accuracy_after)
        history['difficult_accuracies'].append(difficult_acc_after)
        history['teacher_rewards'].append(reward)
        history['actions'].append(action)
        history['topics'].append(action.topic)
        history['difficulties'].append(action.difficulty)
        history['is_reviews'].append(action.is_review)
    
    return history


def plot_comparison(histories: Dict[str, Dict], save_path: str = 'teacher_agent_dev/comparison_all_strategies.png'):
    """
    Create comprehensive comparison plots of all three strategies.
    
    Args:
        histories: Dictionary mapping strategy name to history
                   e.g., {'Random': history1, 'Progressive': history2, 'Teacher': history3}
        save_path: Where to save the plot
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    # Define colors and styles for each strategy
    colors = {
        'Random': '#FF6B6B',      # Red
        'Progressive': '#4ECDC4', # Teal
        'Teacher': '#2ECC71'      # Green (highlight teacher as best)
    }
    
    line_styles = {
        'Random': '--',
        'Progressive': '-.',
        'Teacher': '-'  # Solid line for teacher
    }
    
    line_widths = {
        'Random': 2.0,
        'Progressive': 2.0,
        'Teacher': 3.0  # Thicker line for teacher
    }
    
    # 1. Plot 1: General Accuracy Over Time
    ax = axes[0]
    for name, history in histories.items():
        iterations = history['iterations']
        accuracies = history['student_accuracies']
        
        # Smooth the curve for better visualization
        if len(accuracies) > 50:
            window = 20
            smoothed = np.convolve(accuracies, np.ones(window)/window, mode='same')
            ax.plot(iterations, smoothed, 
                   label=name, 
                   color=colors[name],
                   linestyle=line_styles[name],
                   linewidth=line_widths[name],
                   alpha=0.9)
        else:
            ax.plot(iterations, accuracies, 
                   label=name, 
                   color=colors[name],
                   linestyle=line_styles[name],
                   linewidth=line_widths[name],
                   alpha=0.9)
    
    ax.set_xlabel('Training Iteration', fontsize=12, fontweight='bold')
    ax.set_ylabel('General Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Student Accuracy Comparison: Teacher vs Baselines', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim([0.2, 1.0])
    
    # Add final accuracy annotations
    for name, history in histories.items():
        final_acc = history['student_accuracies'][-1]
        final_iter = history['iterations'][-1]
        ax.annotate(f'{final_acc:.3f}', 
                   xy=(final_iter, final_acc),
                   xytext=(10, 10),
                   textcoords='offset points',
                   fontsize=10,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor=colors[name], alpha=0.3),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    # 2. Plot 2: Difficult Question Accuracy (Most Important!)
    ax = axes[1]
    for name, history in histories.items():
        iterations = history['iterations']
        difficult_accuracies = history['difficult_accuracies']
        
        # Smooth the curve
        if len(difficult_accuracies) > 50:
            window = 20
            smoothed = np.convolve(difficult_accuracies, np.ones(window)/window, mode='same')
            ax.plot(iterations, smoothed, 
                   label=name, 
                   color=colors[name],
                   linestyle=line_styles[name],
                   linewidth=line_widths[name],
                   alpha=0.9)
        else:
            ax.plot(iterations, difficult_accuracies, 
                   label=name, 
                   color=colors[name],
                   linestyle=line_styles[name],
                   linewidth=line_widths[name],
                   alpha=0.9)
    
    ax.set_xlabel('Training Iteration', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy on Difficult Questions', fontsize=12, fontweight='bold')
    ax.set_title('Performance on Difficult Questions (Key Metric)', fontsize=14, fontweight='bold', color='darkred')
    ax.legend(loc='lower right', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim([0.2, 1.0])
    
    # Highlight target accuracy line (75%)
    ax.axhline(y=0.75, color='gray', linestyle=':', linewidth=1, alpha=0.5, label='Target (75%)')
    ax.legend(loc='lower right', fontsize=11, framealpha=0.9)
    
    # Add final accuracy annotations
    for name, history in histories.items():
        final_acc = history['difficult_accuracies'][-1]
        final_iter = history['iterations'][-1]
        ax.annotate(f'{final_acc:.3f}', 
                   xy=(final_iter, final_acc),
                   xytext=(10, 10),
                   textcoords='offset points',
                   fontsize=10,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor=colors[name], alpha=0.3),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    # 3. Plot 3: Learning Speed Comparison (Iterations to reach 75% on difficult)
    ax = axes[2]
    
    target_acc = 0.75
    strategy_stats = {}
    
    for name, history in histories.items():
        difficult_accuracies = history['difficult_accuracies']
        iterations = history['iterations']
        
        # Find when target is reached
        reached_target = False
        target_iteration = len(iterations) - 1
        
        for i, acc in enumerate(difficult_accuracies):
            if acc >= target_acc:
                target_iteration = i
                reached_target = True
                break
        
        strategy_stats[name] = {
            'reached': reached_target,
            'iteration': target_iteration,
            'final_acc': difficult_accuracies[-1]
        }
    
    # Create bar plot
    names = list(strategy_stats.keys())
    iterations_to_target = [
        strategy_stats[n]['iteration'] if strategy_stats[n]['reached'] else len(histories[n]['iterations'])
        for n in names
    ]
    final_accs = [strategy_stats[n]['final_acc'] for n in names]
    
    x = np.arange(len(names))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, iterations_to_target, width, label='Iterations to 75% on Difficult', 
                   color=[colors[n] for n in names], alpha=0.7)
    bars2 = ax.bar(x + width/2, [acc * max(iterations_to_target) for acc in final_accs], width,
                   label='Final Difficult Accuracy (scaled)', 
                   color=[colors[n] for n in names], alpha=0.5)
    
    ax.set_xlabel('Strategy', fontsize=12, fontweight='bold')
    ax.set_ylabel('Iterations / Scaled Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Learning Efficiency: Iterations to Reach Target vs Final Performance', 
                fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    # Add value labels on bars
    for i, (bar1, bar2, name) in enumerate(zip(bars1, bars2, names)):
        height1 = bar1.get_height()
        height2 = bar2.get_height()
        
        # Label for iterations
        if strategy_stats[name]['reached']:
            ax.text(bar1.get_x() + bar1.get_width()/2., height1,
                   f'{int(height1)}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
        else:
            ax.text(bar1.get_x() + bar1.get_width()/2., height1,
                   'Not reached',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Label for final accuracy
        ax.text(bar2.get_x() + bar2.get_width()/2., height2,
               f'{final_accs[i]:.2f}',
               ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Saved comparison plot to {save_path}")
    plt.close()
    
    # Print summary statistics
    print("\n" + "=" * 70)
    print("STRATEGY COMPARISON SUMMARY")
    print("=" * 70)
    for name, stats in strategy_stats.items():
        status = "✅ Reached" if stats['reached'] else "❌ Not reached"
        print(f"{name:15s} | {status:15s} | Iterations: {stats['iteration']:4d} | Final Acc: {stats['final_acc']:.3f}")
    print("=" * 70)


if __name__ == "__main__":
    print("=" * 70)
    print("COMPARING THREE TRAINING STRATEGIES")
    print("=" * 70)
    print("\n1. Random: Random questions until student can pass difficult")
    print("2. Progressive: Easy → Medium → Hard within each family")
    print("3. Teacher: RL teacher agent learns optimal curriculum")
    print("\n" + "=" * 70 + "\n")
    
    num_iterations = 500
    seed = 42
    
    # Train all three strategies
    print("Training Random Strategy...")
    history_random = train_strategy_random(num_iterations=num_iterations, seed=seed)
    
    print("\nTraining Progressive Strategy...")
    history_progressive = train_strategy_progressive(num_iterations=num_iterations, seed=seed)
    
    print("\nTraining Teacher Strategy...")
    history_teacher = train_strategy_teacher(num_iterations=num_iterations, seed=seed)
    
    # Create comparison plots
    print("\nGenerating comparison plots...")
    histories = {
        'Random': history_random,
        'Progressive': history_progressive,
        'Teacher': history_teacher
    }
    
    plot_comparison(histories, save_path='comparison_all_strategies.png')
    
    print("\n✅ Comparison complete! Check 'comparison_all_strategies.png'")

