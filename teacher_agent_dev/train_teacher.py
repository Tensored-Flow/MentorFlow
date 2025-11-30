"""Main training loop for Teacher Agent system."""

import numpy as np
from typing import Dict, Tuple
from interfaces import Task

from mock_student import MockStudentAgent
from mock_task_generator import MockTaskGenerator
from teacher_agent import TeacherAgent, compute_reward


def train_teacher(num_iterations: int = 500, verbose: bool = True, seed: int = 42) -> Tuple[Dict, TeacherAgent, MockStudentAgent]:
    """
    Train teacher agent with mock student.
    
    Args:
        num_iterations: Number of training iterations
        verbose: Whether to print progress
        seed: Random seed
    
    Returns:
        Tuple of (history dict, teacher agent, student agent)
    """
    # Initialize components
    teacher = TeacherAgent(exploration_bonus=2.0)
    student = MockStudentAgent(learning_rate=0.15, forgetting_rate=0.05, seed=seed)
    generator = MockTaskGenerator(seed=seed)
    
    # Create evaluation set (held-out tasks for measuring student performance)
    eval_tasks = []
    for topic in generator.get_available_topics():
        for _ in range(3):  # 3 tasks per topic
            eval_tasks.append(generator.generate_task(topic, 'medium'))
    
    if verbose:
        print("=" * 70)
        print("TEACHER AGENT TRAINING")
        print("=" * 70)
        print(f"Iterations: {num_iterations}")
        print(f"Evaluation tasks: {len(eval_tasks)}")
        print(f"Action space: {teacher.num_actions} actions")
        print("=" * 70)
    
    # Track metrics
    history = {
        'iterations': [],
        'student_accuracies': [],
        'teacher_rewards': [],
        'actions': [],
        'topics': [],
        'difficulties': [],
        'is_reviews': []
    }
    
    for iteration in range(num_iterations):
        # 1. Get student state
        student_state = student.get_state()
        
        # 2. Teacher selects action
        action = teacher.select_action(student_state)
        
        # 3. Generate task
        # For reviews, use same topic but maybe different difficulty
        if action.is_review:
            # Review: use same topic, medium difficulty
            task = generator.generate_task(action.topic, 'medium')
        else:
            # New material: use specified topic and difficulty
            task = generator.generate_task(action.topic, action.difficulty)
        
        # 4. Evaluate student BEFORE learning
        accuracy_before = student.evaluate(eval_tasks)
        
        # 5. Student learns from task
        was_correct = student.learn(task)
        
        # 6. Evaluate student AFTER learning
        accuracy_after = student.evaluate(eval_tasks)
        
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
        history['teacher_rewards'].append(reward)
        history['actions'].append(action)
        history['topics'].append(action.topic)
        history['difficulties'].append(action.difficulty)
        history['is_reviews'].append(action.is_review)
        
        # 11. Print progress
        if verbose and (iteration % 50 == 0 or iteration == num_iterations - 1):
            window = min(50, iteration + 1)
            recent_rewards = history['teacher_rewards'][-window:]
            avg_reward = np.mean(recent_rewards) if recent_rewards else 0.0
            
            print(f"Iteration {iteration:3d} | "
                  f"Student Acc: {accuracy_after:.3f} | "
                  f"Avg Reward: {avg_reward:.3f} | "
                  f"Action: {action.topic[:3]}-{action.difficulty[:2]}-{'R' if action.is_review else 'N'}")
    
    if verbose:
        print("=" * 70)
        print(f"Final accuracy: {history['student_accuracies'][-1]:.3f}")
        print(f"Average reward: {np.mean(history['teacher_rewards']):.3f}")
        print("=" * 70)
    
    return history, teacher, student


def train_baseline_random(num_iterations: int = 500, seed: int = 42) -> Dict:
    """Train with random teacher (baseline)."""
    import random
    rng = random.Random(seed)
    
    student = MockStudentAgent(learning_rate=0.15, forgetting_rate=0.05, seed=seed)
    generator = MockTaskGenerator(seed=seed)
    
    topics = generator.get_available_topics()
    difficulties = generator.get_available_difficulties()
    
    eval_tasks = [
        generator.generate_task(topic, 'medium')
        for topic in topics
        for _ in range(3)
    ]
    
    history = {
        'iterations': [],
        'student_accuracies': [],
        'teacher_rewards': [],
        'actions': [],
        'topics': [],
        'difficulties': [],
        'is_reviews': []
    }
    
    for iteration in range(num_iterations):
        # Random action
        topic = rng.choice(topics)
        difficulty = rng.choice(difficulties)
        is_review = rng.random() < 0.3  # 30% chance of review
        
        task = generator.generate_task(topic, 'medium' if is_review else difficulty)
        
        accuracy_before = student.evaluate(eval_tasks)
        student.learn(task)
        accuracy_after = student.evaluate(eval_tasks)
        
        reward = compute_reward(accuracy_before, accuracy_after, difficulty, is_review)
        
        student.advance_time(1.0)
        
        history['iterations'].append(iteration)
        history['student_accuracies'].append(accuracy_after)
        history['teacher_rewards'].append(reward)
        history['topics'].append(topic)
        history['difficulties'].append(difficulty)
        history['is_reviews'].append(is_review)
    
    return history


def train_baseline_fixed(num_iterations: int = 500, seed: int = 42) -> Dict:
    """Train with fixed curriculum (easy→medium→hard, sequential topics)."""
    student = MockStudentAgent(learning_rate=0.15, forgetting_rate=0.05, seed=seed)
    generator = MockTaskGenerator(seed=seed)
    
    topics = generator.get_available_topics()
    difficulties = ['easy', 'medium', 'hard']
    
    eval_tasks = [
        generator.generate_task(topic, 'medium')
        for topic in topics
        for _ in range(3)
    ]
    
    history = {
        'iterations': [],
        'student_accuracies': [],
        'teacher_rewards': [],
        'actions': [],
        'topics': [],
        'difficulties': [],
        'is_reviews': []
    }
    
    # Fixed curriculum: cycle through topics, increase difficulty over time
    phase_length = num_iterations // (len(topics) * len(difficulties))
    
    for iteration in range(num_iterations):
        # Determine phase
        phase = iteration // phase_length
        topic_idx = (phase // len(difficulties)) % len(topics)
        diff_idx = phase % len(difficulties)
        
        topic = topics[topic_idx]
        difficulty = difficulties[diff_idx]
        
        task = generator.generate_task(topic, difficulty)
        
        accuracy_before = student.evaluate(eval_tasks)
        student.learn(task)
        accuracy_after = student.evaluate(eval_tasks)
        
        reward = compute_reward(accuracy_before, accuracy_after, difficulty, False)
        
        student.advance_time(1.0)
        
        history['iterations'].append(iteration)
        history['student_accuracies'].append(accuracy_after)
        history['teacher_rewards'].append(reward)
        history['topics'].append(topic)
        history['difficulties'].append(difficulty)
        history['is_reviews'].append(False)
    
    return history


if __name__ == "__main__":
    # Train teacher agent
    print("\n" + "=" * 70)
    print("TRAINING TEACHER AGENT")
    print("=" * 70)
    history, teacher, student = train_teacher(num_iterations=500, verbose=True)
    
    # Print statistics
    stats = teacher.get_statistics()
    print(f"\nTeacher Statistics:")
    print(f"  Total actions tried: {stats['total_pulls']}")
    print(f"  Unique actions: {np.sum(stats['action_counts'] > 0)}/{stats['total_pulls']}")

