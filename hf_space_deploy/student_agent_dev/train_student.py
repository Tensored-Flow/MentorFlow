"""
Main training script for student agent.

Integrates student with mock teacher/task generator and generates
comprehensive visualizations.
"""

import torch
from student_agent import StudentAgent
from student_metrics import StudentMetrics
from mock_teacher import MockTeacherAgent
from mock_task_generator import MockTaskGenerator
from visualize_student import create_comprehensive_report


def compute_teacher_reward(
    accuracy_before: float,
    accuracy_after: float,
    difficulty: str,
    is_review: bool
) -> float:
    """Reward function for teacher (shared with teacher agent)."""
    improvement = accuracy_after - accuracy_before
    
    difficulty_bonus = {'easy': 0.5, 'medium': 1.0, 'hard': 2.0}.get(difficulty, 1.0)
    review_bonus = 1.0 if (is_review and improvement > 0) else 0.0
    review_penalty = -0.5 if (is_review and accuracy_after > 0.9) else 0.0
    
    return improvement + difficulty_bonus + review_bonus + review_penalty


def train_student(
    num_iterations: int = 500,
    device: str = 'cpu',
    learning_rate: float = 5e-5,
    retention_constant: float = 80.0,
    verbose: bool = True
):
    """
    Train student agent with mock teacher and task generator.
    
    Args:
        num_iterations: Number of training iterations
        device: 'cpu' or 'cuda'
        learning_rate: Student LM learning rate
        retention_constant: Memory decay rate (higher = slower forgetting)
        verbose: Print progress
        
    Returns:
        Tuple of (metrics, student, teacher, generator)
    """
    # Initialize components
    if verbose:
        print("Initializing student agent...")
    
    student = StudentAgent(
        learning_rate=learning_rate,
        retention_constant=retention_constant,
        device=device
    )
    
    teacher = MockTeacherAgent()
    generator = MockTaskGenerator()
    
    # Create evaluation set (held-out for measuring progress)
    eval_tasks = []
    for topic in generator.get_available_topics():
        for difficulty in ['easy', 'medium', 'hard']:
            for _ in range(2):  # 2 tasks per (topic, difficulty)
                eval_tasks.append(generator.generate_task(topic, difficulty))
    
    if verbose:
        print(f"Created evaluation set: {len(eval_tasks)} tasks")
        print(f"Training for {num_iterations} iterations...\n")
    
    # Initialize metrics tracker
    metrics = StudentMetrics()
    
    # Training loop
    for iteration in range(num_iterations):
        # 1. Get student state
        student_state = student.get_state()
        
        # 2. Teacher selects action
        action = teacher.select_action(student_state)
        
        # 3. Generate task
        task = generator.generate_task(action.topic, action.difficulty)
        
        # 4. Evaluate BEFORE learning
        accuracy_before = student.evaluate(eval_tasks)
        
        # 5. Student learns from task
        was_correct = student.learn(task)
        
        # 6. Evaluate AFTER learning
        accuracy_after = student.evaluate(eval_tasks)
        
        # 7. Compute teacher reward (for compatibility with teacher agent)
        reward = compute_teacher_reward(
            accuracy_before, accuracy_after,
            action.difficulty, action.is_review
        )
        
        # 8. Update teacher (mock doesn't use this)
        teacher.update(action, reward)
        
        # 9. Time passes (for forgetting)
        student.advance_time(1.0)
        
        # 10. Log metrics
        topic_accuracies = {
            topic: student.memory.get_effective_skill(topic)
            for topic in student.topic_base_skills
        }
        
        retention_factors = {
            topic: student.memory.get_retention_factor(topic)
            for topic in student.topic_base_skills
        }
        
        metrics.log_iteration(
            iteration=iteration,
            overall_acc=accuracy_after,
            topic_accs=topic_accuracies,
            task=task,
            correct=was_correct,
            retention_factors=retention_factors
        )
        
        # 11. Print progress
        if verbose and iteration % 50 == 0:
            avg_acc = accuracy_after
            topics_practiced = len(student.topic_base_skills)
            print(f"Iteration {iteration:3d} | "
                  f"Accuracy: {avg_acc:.3f} | "
                  f"Topics: {topics_practiced} | "
                  f"Correct: {'✓' if was_correct else '✗'}")
    
    if verbose:
        print("\n✅ Training complete!")
    
    return metrics, student, teacher, generator


def main():
    """Main entry point."""
    # Check if CUDA available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    # Train student
    metrics, student, teacher, generator = train_student(
        num_iterations=500,
        device=device,
        learning_rate=5e-5,
        retention_constant=80.0,
        verbose=True
    )
    
    # Generate visualizations
    create_comprehensive_report(metrics, output_dir='student_visualizations')
    
    # Save model checkpoint
    student.save('student_checkpoint.pt')
    if verbose:
        print("\n💾 Saved student checkpoint to student_checkpoint.pt")


if __name__ == "__main__":
    main()

