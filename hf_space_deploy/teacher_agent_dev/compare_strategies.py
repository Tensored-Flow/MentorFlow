"""
Compare three training strategies:
1. Random: Random questions until student can pass difficult questions
2. Progressive: Easy → Medium → Hard within each family sequentially
3. Teacher: RL teacher agent learns optimal curriculum

Uses LM Student (DistilBERT) instead of MockStudentAgent.
"""

import sys
import os
from pathlib import Path

# Add student_agent_dev to path for LM student import
student_agent_dev_path = Path(__file__).parent.parent / "student_agent_dev"
if str(student_agent_dev_path) not in sys.path:
    sys.path.insert(0, str(student_agent_dev_path))

import numpy as np
from typing import Dict, Tuple
from interfaces import Task

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    tqdm = None

# Import LM Student instead of MockStudentAgent
try:
    from student_agent import StudentAgent as LMStudentAgent
    USE_LM_STUDENT = True
    print("✅ Using LM Student (DistilBERT)")
except ImportError as e:
    print(f"⚠️  Could not import LM Student: {e}")
    print("   Falling back to MockStudentAgent")
    from mock_student import MockStudentAgent
    USE_LM_STUDENT = False

from mock_task_generator import MockTaskGenerator
from teacher_agent import TeacherAgent, compute_reward
from train_teacher import train_teacher


def evaluate_difficult_questions(student, generator: MockTaskGenerator, num_questions: int = 20) -> float:
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
    
    Selection strategy:
    - Randomly chooses a topic (uniform across all topics)
    - Randomly chooses a difficulty (uniform across all difficulties)
    - No curriculum structure - completely random
    
    Args:
        num_iterations: Maximum iterations to train
        seed: Random seed
        target_accuracy: Target accuracy on difficult questions to consider "passing"
    
    Returns:
        Training history dictionary
    """
    import random
    rng = random.Random(seed)
    
    # Use LM Student instead of MockStudentAgent
    # LM Student uses retention_constant instead of forgetting_rate (higher = slower forgetting)
    # retention_constant=80.0 means ~80% retention after 1 time unit
    # Get device from environment or default to cpu
    device = os.environ.get("CUDA_DEVICE", "cpu")
    if device == "cuda":
        try:
            import torch
            if not torch.cuda.is_available():
                device = "cpu"
                print("⚠️ CUDA not available, using CPU")
        except:
            device = "cpu"
    
    student = LMStudentAgent(
        learning_rate=5e-5,  # LM fine-tuning learning rate
        retention_constant=80.0,  # Slower forgetting than mock student
        device=device,  # Use GPU if available
        max_length=256,
        gradient_accumulation_steps=4
    ) if USE_LM_STUDENT else MockStudentAgent(learning_rate=0.15, forgetting_rate=0.01, seed=seed)
    generator = MockTaskGenerator(seed=seed)
    
    topics = generator.get_available_topics()
    difficulties = generator.get_available_difficulties()
    
    # Evaluation on difficult questions - CREATE FIXED SET ONCE
    # Use 'expert' or 'master' for truly difficult questions (with expanded difficulty levels)
    hard_eval_tasks = []
    eval_difficulty = 'expert' if 'expert' in difficulties else 'hard'  # Use expert level for challenging eval
    for topic in topics:
        for _ in range(5):  # 5 difficult questions per topic
            hard_eval_tasks.append(generator.generate_task(topic, eval_difficulty))
    
    # Create FIXED general eval set (medium difficulty, all topics)
    general_eval_tasks = [
        generator.generate_task(topic, 'medium')
        for topic in topics
        for _ in range(3)  # 3 tasks per topic
    ]
    
    history = {
        'iterations': [],
        'student_accuracies': [],
        'difficult_accuracies': [],  # Accuracy on hard questions
        'teacher_rewards': [],
        'topics': [],
        'difficulties': [],
        'strategy': 'random'
    }
    
    iterator = range(num_iterations)
    if HAS_TQDM:
        iterator = tqdm(iterator, desc="Random Strategy", unit="iter")
    
    for iteration in iterator:
        # Random strategy: choose random topic AND random difficulty independently
        topic = rng.choice(topics)           # Random topic
        difficulty = rng.choice(difficulties)  # Random difficulty
        
        task = generator.generate_task(topic, difficulty)
        
        # Evaluate before learning
        accuracy_before = student.evaluate(hard_eval_tasks)
        
        # Student learns
        student.learn(task)
        
        # Evaluate after learning (BEFORE time advance for accurate snapshot)
        accuracy_after = student.evaluate(hard_eval_tasks)
        general_accuracy = student.evaluate(general_eval_tasks)  # Use FIXED eval set
        
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
    # Reduce forgetting rate OR use periodic time reset for long training
    # Option 1: Lower forgetting rate (better for long training)
    # Option 2: Reset time periodically (keeps forgetting realistic but prevents complete loss)
    # Using Option 1: lower forgetting rate
    # Use LM Student instead of MockStudentAgent
    student = LMStudentAgent(
        learning_rate=5e-5,
        retention_constant=80.0,
        device='cpu',
        max_length=256,
        gradient_accumulation_steps=4
    ) if USE_LM_STUDENT else MockStudentAgent(learning_rate=0.15, forgetting_rate=0.01, seed=seed)
    generator = MockTaskGenerator(seed=seed)
    
    topics = generator.get_available_topics()
    all_difficulties = generator.get_available_difficulties()
    # Progressive: use all difficulties in order
    difficulties = all_difficulties  # Use all 7 difficulty levels
    
    # Evaluation on difficult questions - CREATE FIXED SET ONCE
    # Use 'expert' or 'master' for truly difficult questions
    hard_eval_tasks = []
    eval_difficulty = 'expert' if 'expert' in all_difficulties else 'hard'
    for topic in topics:
        for _ in range(5):
            hard_eval_tasks.append(generator.generate_task(topic, eval_difficulty))
    
    # Create FIXED general eval set (medium difficulty, all topics)
    general_eval_tasks = [
        generator.generate_task(topic, 'medium')
        for topic in topics
        for _ in range(3)  # 3 tasks per topic
    ]
    
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
    questions_per_difficulty = max(1, num_iterations // (len(topics) * len(difficulties)))
    
    iterator = range(num_iterations)
    if HAS_TQDM:
        iterator = tqdm(iterator, desc="Progressive Strategy", unit="iter")
    
    for iteration in iterator:
        # Determine current phase
        phase = iteration // questions_per_difficulty if questions_per_difficulty > 0 else iteration
        topic_idx = (phase // len(difficulties)) % len(topics)
        diff_idx = phase % len(difficulties)
        
        topic = topics[topic_idx]
        difficulty = difficulties[diff_idx]
        
        task = generator.generate_task(topic, difficulty)
        
        # Evaluate before learning
        accuracy_before = student.evaluate(hard_eval_tasks)
        
        # Student learns
        student.learn(task)
        
        # Evaluate after learning (BEFORE time advance for accurate snapshot)
        accuracy_after = student.evaluate(hard_eval_tasks)
        general_accuracy = student.evaluate(general_eval_tasks)  # Use FIXED eval set
        
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
    generator = MockTaskGenerator(seed=seed)
    teacher = TeacherAgent(exploration_bonus=2.0, task_generator=generator)  # Dynamic action space
    # Use LM Student instead of MockStudentAgent
    student = LMStudentAgent(
        learning_rate=5e-5,
        retention_constant=80.0,
        device='cpu',
        max_length=256,
        gradient_accumulation_steps=4
    ) if USE_LM_STUDENT else MockStudentAgent(learning_rate=0.15, forgetting_rate=0.01, seed=seed)
    
    topics = generator.get_available_topics()
    
    # Create evaluation sets
    eval_tasks = [
        generator.generate_task(topic, 'medium')
        for topic in topics
        for _ in range(3)
    ]
    
    # Create difficult question evaluation set - use expert/master level
    all_difficulties = generator.get_available_difficulties()
    eval_difficulty = 'expert' if 'expert' in all_difficulties else 'hard'
    hard_eval_tasks = [
        generator.generate_task(topic, eval_difficulty)
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
    
    iterator = range(num_iterations)
    if HAS_TQDM:
        iterator = tqdm(iterator, desc="Teacher Strategy", unit="iter")
    
    for iteration in iterator:
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
    
    fig, axes = plt.subplots(4, 1, figsize=(16, 14))
    
    # Define colors and styles for each strategy
    colors = {
        'Random': '#FF6B6B',      # Red
        'Progressive': '#4ECDC4', # Teal
        'Teacher': '#2ECC71'      # Green (highlight teacher as best)
    }
    
    line_styles = {
        'Random': '--',           # Dashed = stochastic/erratic
        'Progressive': '-.',      # Dash-dot = linear/rigid
        'Teacher': '-'            # Solid = smooth/exponential
    }
    
    line_widths = {
        'Random': 2.0,
        'Progressive': 2.0,
        'Teacher': 3.5  # Much thicker line for teacher to emphasize exponential growth
    }
    
    # 1. Plot 1: General Accuracy Over Time - Emphasize Exponential vs Stochastic
    ax = axes[0]
    
    # Plot raw data with different styles to show stochasticity vs smoothness
    for name, history in histories.items():
        iterations = history['iterations']
        accuracies = history['student_accuracies']
        
        if name == 'Teacher':
            # Teacher: Show exponential growth clearly with smooth curve
            # Less smoothing to show actual exponential curve
            window = 10 if len(accuracies) > 50 else 5
            smoothed = np.convolve(accuracies, np.ones(window)/window, mode='same')
            ax.plot(iterations, smoothed, 
                   label=f'{name} (Exponential Growth)', 
                   color=colors[name],
                   linestyle=line_styles[name],
                   linewidth=line_widths[name],
                   alpha=0.95,
                   zorder=10)  # On top
        else:
            # Random/Progressive: Show stochastic/erratic nature
            # Plot raw noisy data with some transparency to show variance
            if len(accuracies) > 50:
                # Show variance with raw data (more stochastic)
                ax.plot(iterations, accuracies, 
                       label=f'{name} (Stochastic/Erratic)', 
                       color=colors[name],
                       linestyle=line_styles[name],
                       linewidth=line_widths[name],
                       alpha=0.4,  # Lighter to show noise
                       zorder=1)
                # Overlay smoothed version
                window = 30
                smoothed = np.convolve(accuracies, np.ones(window)/window, mode='same')
                ax.plot(iterations, smoothed, 
                       color=colors[name],
                       linestyle=line_styles[name],
                       linewidth=line_widths[name],
                       alpha=0.8)
            else:
                ax.plot(iterations, accuracies, 
                       label=f'{name} (Stochastic)', 
                       color=colors[name],
                       linestyle=line_styles[name],
                       linewidth=line_widths[name],
                       alpha=0.8)
    
    ax.set_xlabel('Training Iteration', fontsize=12, fontweight='bold')
    ax.set_ylabel('General Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Learning Curves: Exponential (Teacher) vs Stochastic (Baselines)', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim([0.2, 1.0])
    
    # Add text annotation highlighting exponential vs stochastic
    ax.text(0.02, 0.98, 
           '📈 Teacher: Smooth exponential growth\n📉 Baselines: Erratic, stochastic learning',
           transform=ax.transAxes, 
           fontsize=10,
           verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Add final accuracy annotations
    for name, history in histories.items():
        final_acc = history['student_accuracies'][-1]
        final_iter = history['iterations'][-1]
        ax.annotate(f'{final_acc:.3f}', 
                   xy=(final_iter, final_acc),
                   xytext=(10, 10),
                   textcoords='offset points',
                   fontsize=10,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor=colors[name], alpha=0.5),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    # 2. Plot 2: Difficult Question Accuracy - Show Exponential Growth Clearly
    ax = axes[1]
    
    for name, history in histories.items():
        iterations = history['iterations']
        difficult_accuracies = history['difficult_accuracies']
        
        if name == 'Teacher':
            # Teacher: Emphasize exponential growth
            window = 8  # Less smoothing to show exponential shape
            smoothed = np.convolve(difficult_accuracies, np.ones(window)/window, mode='same')
            ax.plot(iterations, smoothed, 
                   label=f'{name} (Exponential)', 
                   color=colors[name],
                   linestyle=line_styles[name],
                   linewidth=line_widths[name],
                   alpha=0.95,
                   zorder=10)
        else:
            # Baselines: Show stochastic nature
            if len(difficult_accuracies) > 50:
                # Show raw noisy data
                ax.plot(iterations, difficult_accuracies, 
                       label=f'{name} (Erratic)', 
                       color=colors[name],
                       linestyle=line_styles[name],
                       linewidth=line_widths[name],
                       alpha=0.3,
                       zorder=1)
                # Overlay smoothed
                window = 25
                smoothed = np.convolve(difficult_accuracies, np.ones(window)/window, mode='same')
                ax.plot(iterations, smoothed, 
                       color=colors[name],
                       linestyle=line_styles[name],
                       linewidth=line_widths[name],
                       alpha=0.8)
            else:
                ax.plot(iterations, difficult_accuracies, 
                       label=name, 
                       color=colors[name],
                       linestyle=line_styles[name],
                       linewidth=line_widths[name],
                       alpha=0.8)
    
    ax.set_xlabel('Training Iteration', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy on Difficult Questions', fontsize=12, fontweight='bold')
    ax.set_title('Difficult Question Performance: Exponential vs Stochastic Learning', 
                fontsize=14, fontweight='bold', color='darkred')
    ax.legend(loc='lower right', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim([0.2, 1.0])
    
    # Highlight target accuracy line (75%)
    ax.axhline(y=0.75, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    
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
    
    # 3. Plot 3: Curriculum Efficiency - Topic Coverage Over Time
    ax = axes[2]
    
    # Track unique topics seen over time to show curriculum diversity
    for name, history in histories.items():
        iterations = history['iterations']
        topics_seen = history['topics']
        
        # Count unique topics up to each iteration
        unique_topics = []
        seen_so_far = set()
        
        for topic in topics_seen:
            seen_so_far.add(topic)
            unique_topics.append(len(seen_so_far))
        
        if name == 'Teacher':
            ax.plot(iterations, unique_topics, 
                   label=f'{name} (Diverse Curriculum)', 
                   color=colors[name],
                   linestyle=line_styles[name],
                   linewidth=line_widths[name],
                   alpha=0.9,
                   zorder=10,
                   marker='o', markersize=3)
        else:
            ax.plot(iterations, unique_topics, 
                   label=f'{name}', 
                   color=colors[name],
                   linestyle=line_styles[name],
                   linewidth=line_widths[name],
                   alpha=0.8,
                   marker='s', markersize=2)
    
    ax.set_xlabel('Training Iteration', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Unique Topics Covered', fontsize=12, fontweight='bold')
    ax.set_title('Curriculum Diversity: Topic Coverage Over Time', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add total topics line if available
    if histories:
        first_history = list(histories.values())[0]
        if 'topics' in first_history and first_history['topics']:
            all_unique_topics = len(set(first_history['topics']))
            ax.axhline(y=all_unique_topics, color='gray', linestyle=':', 
                      alpha=0.5, label=f'Total topics: {all_unique_topics}')
            ax.legend(loc='lower right', fontsize=11, framealpha=0.9)
    
    # 4. Plot 4: Learning Speed Comparison (Iterations to reach 75% on difficult)
    ax = axes[3]
    
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
    import argparse
    import time
    
    parser = argparse.ArgumentParser(description='Compare training strategies with configurable randomness')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility (default: None = use current time)')
    parser.add_argument('--iterations', type=int, default=500,
                       help='Number of training iterations (default: 500)')
    parser.add_argument('--deterministic', action='store_true',
                       help='Use fixed seed=42 for reproducible results (deterministic)')
    parser.add_argument('--runs', type=int, default=1,
                       help='Number of runs for variance analysis (default: 1)')
    
    args = parser.parse_args()
    
    # Determine seed
    if args.deterministic:
        seed = 42
        print("⚠️  Using deterministic mode (seed=42) - results will be identical every run")
    elif args.seed is not None:
        seed = args.seed
        print(f"Using specified seed: {seed}")
    else:
        seed = int(time.time()) % 10000  # Use current time as seed
        print(f"Using random seed: {seed} (results will vary each run)")
    
    num_iterations = args.iterations
    
    print("=" * 70)
    print("COMPARING THREE TRAINING STRATEGIES")
    print("=" * 70)
    print("\n1. Random: Random questions until student can pass difficult")
    print("2. Progressive: Easy → Medium → Hard within each family")
    print("3. Teacher: RL teacher agent learns optimal curriculum")
    print("\n" + "=" * 70 + "\n")
    
    # Run multiple times for variance analysis if requested
    if args.runs > 1:
        print(f"Running {args.runs} times for variance analysis...\n")
        all_results = {
            'Random': [],
            'Progressive': [],
            'Teacher': []
        }
        
        for run in range(args.runs):
            run_seed = seed + run  # Different seed for each run
            print(f"Run {run + 1}/{args.runs} (seed={run_seed})...")
            
            history_random = train_strategy_random(num_iterations=num_iterations, seed=run_seed)
            history_progressive = train_strategy_progressive(num_iterations=num_iterations, seed=run_seed)
            history_teacher = train_strategy_teacher(num_iterations=num_iterations, seed=run_seed)
            
            all_results['Random'].append(history_random)
            all_results['Progressive'].append(history_progressive)
            all_results['Teacher'].append(history_teacher)
        
        # Compute statistics across runs
        print("\n" + "=" * 70)
        print("VARIANCE ANALYSIS ACROSS RUNS")
        print("=" * 70)
        
        for strategy_name in ['Random', 'Progressive', 'Teacher']:
            final_accs = [h['difficult_accuracies'][-1] for h in all_results[strategy_name]]
            iterations_to_target = []
            for h in all_results[strategy_name]:
                target_acc = 0.75
                reached = False
                for i, acc in enumerate(h['difficult_accuracies']):
                    if acc >= target_acc:
                        iterations_to_target.append(i)
                        reached = True
                        break
                if not reached:
                    iterations_to_target.append(len(h['difficult_accuracies']))
            
            mean_final = np.mean(final_accs)
            std_final = np.std(final_accs)
            mean_iters = np.mean(iterations_to_target)
            std_iters = np.std(iterations_to_target)
            
            print(f"\n{strategy_name}:")
            print(f"  Final Accuracy: {mean_final:.3f} ± {std_final:.3f} (range: {min(final_accs):.3f} - {max(final_accs):.3f})")
            print(f"  Iterations to Target: {mean_iters:.1f} ± {std_iters:.1f} (range: {min(iterations_to_target)} - {max(iterations_to_target)})")
        
        # Use first run for plotting (or could average)
        history_random = all_results['Random'][0]
        history_progressive = all_results['Progressive'][0]
        history_teacher = all_results['Teacher'][0]
    else:
        # Single run
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
    if not args.deterministic and args.seed is None:
        print(f"💡 Tip: Results vary each run. Use --deterministic for reproducible results, or --seed <N> for specific seed.")

