"""
Diagnose why accuracy drops at the end of training.

Issues to investigate:
1. Evaluation task generation (are they consistent?)
2. Forgetting over time
3. Evaluation timing (before/after learning, before/after time advance)
"""

import numpy as np
from mock_student import MockStudentAgent
from mock_task_generator import MockTaskGenerator

def diagnose_evaluation():
    """Check if evaluation tasks are consistent."""
    print("=" * 70)
    print("DIAGNOSING ACCURACY DROP")
    print("=" * 70)
    
    generator = MockTaskGenerator(seed=42)
    student = MockStudentAgent(learning_rate=0.15, forgetting_rate=0.05, seed=42)
    
    topics = generator.get_available_topics()
    
    # Create FIXED eval set
    fixed_eval_tasks = [
        generator.generate_task(topic, 'medium')
        for topic in topics
        for _ in range(3)
    ]
    
    print(f"\n1. Fixed eval set created: {len(fixed_eval_tasks)} tasks")
    
    # Check if regenerating tasks gives same tasks
    print("\n2. Checking task consistency...")
    task1 = generator.generate_task('history', 'medium')
    generator2 = MockTaskGenerator(seed=42)
    task2 = generator2.generate_task('history', 'medium')
    print(f"   Same seed, same topic: {'SAME' if task1.question == task2.question else 'DIFFERENT'}")
    
    # Simulate training and track accuracy
    print("\n3. Simulating training with FIXED eval set...")
    accuracies = []
    time_points = []
    
    for iteration in range(500):
        # Random learning
        import random
        rng = random.Random(42 + iteration)
        topic = rng.choice(topics)
        difficulty = rng.choice(['easy', 'medium', 'hard'])
        
        task = generator.generate_task(topic, difficulty)
        student.learn(task)
        student.advance_time(1.0)
        
        # Evaluate on FIXED set
        if iteration % 50 == 0:
            acc = student.evaluate(fixed_eval_tasks)
            accuracies.append(acc)
            time_points.append(student.current_time)
            print(f"   Iteration {iteration:3d}, Time: {student.current_time:5.1f}, Acc: {acc:.3f}")
    
    print(f"\n   Accuracy trend: {accuracies[0]:.3f} → {accuracies[-1]:.3f}")
    
    # Now check what happens with REGENERATED eval tasks
    print("\n4. Simulating with REGENERATED eval tasks each time...")
    student2 = MockStudentAgent(learning_rate=0.15, forgetting_rate=0.05, seed=42)
    generator2 = MockTaskGenerator(seed=42)
    accuracies2 = []
    
    for iteration in range(500):
        topic = rng.choice(topics)
        difficulty = rng.choice(['easy', 'medium', 'hard'])
        
        task = generator2.generate_task(topic, difficulty)
        student2.learn(task)
        student2.advance_time(1.0)
        
        if iteration % 50 == 0:
            # Regenerate eval tasks
            new_eval_tasks = [
                generator2.generate_task(t, 'medium')
                for t in topics
                for _ in range(3)
            ]
            acc = student2.evaluate(new_eval_tasks)
            accuracies2.append(acc)
    
    print(f"\n   Accuracy trend: {accuracies2[0]:.3f} → {accuracies2[-1]:.3f}")
    
    # Check forgetting effect
    print("\n5. Checking forgetting effect...")
    student3 = MockStudentAgent(learning_rate=0.15, forgetting_rate=0.05, seed=42)
    generator3 = MockTaskGenerator(seed=42)
    
    # Train intensively
    for _ in range(100):
        for topic in topics:
            task = generator3.generate_task(topic, 'easy')
            student3.learn(task)
    
    # Evaluate immediately
    eval_tasks = [generator3.generate_task(t, 'medium') for t in topics for _ in range(3)]
    acc_before = student3.evaluate(eval_tasks)
    
    # Advance time significantly
    student3.advance_time(100.0)
    acc_after = student3.evaluate(eval_tasks)
    
    print(f"   After intensive training: {acc_before:.3f}")
    print(f"   After 100 time units pass: {acc_after:.3f}")
    print(f"   Forgetting: {acc_before - acc_after:.3f}")
    
    # Check retention formula
    print("\n6. Retention calculation at different time points:")
    base_skill = 1.0  # Perfect skill
    forgetting_rate = 0.05
    
    for time in [0, 50, 100, 200, 500]:
        retention = np.exp(-forgetting_rate * time)
        effective_skill = base_skill * retention
        accuracy = 0.25 + 0.75 * effective_skill
        print(f"   Time={time:3d}: retention={retention:.3f}, accuracy={accuracy:.3f}")

if __name__ == "__main__":
    diagnose_evaluation()

