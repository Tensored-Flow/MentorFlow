#!/usr/bin/env python3
"""
Diagnose why Random strategy outperforms Teacher strategy.
Analyzes task coherence, reward function, and teacher behavior.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "teacher_agent_dev"))

from mock_task_generator import MockTaskGenerator
from mock_student import MockStudentAgent
from teacher_agent import TeacherAgent, compute_reward
from interfaces import Task, StudentState

def analyze_task_coherence():
    """Analyze if tasks are coherent (build on each other) or independent."""
    print("="*70)
    print("TASK COHERENCE ANALYSIS")
    print("="*70)
    
    generator = MockTaskGenerator(seed=42)
    
    # Generate multiple tasks for the same topic
    topic = 'history'
    print(f"\nGenerating 5 tasks for topic '{topic}' at difficulty 'medium':")
    print("-" * 70)
    
    tasks = []
    for i in range(5):
        task = generator.generate_task(topic, 'medium')
        tasks.append(task)
        print(f"\nTask {i+1}:")
        print(f"  Passage: {task.passage[:80]}...")
        print(f"  Question: {task.question}")
        print(f"  Task ID: {task.task_id}")
    
    print("\n" + "="*70)
    print("COHERENCE CHECK:")
    print("="*70)
    
    # Check if tasks reference each other
    passages = [t.passage for t in tasks]
    questions = [t.question for t in tasks]
    
    # Look for references between tasks
    coherent = False
    for i, task1 in enumerate(tasks):
        for j, task2 in enumerate(tasks):
            if i != j:
                # Check if task2 references concepts from task1
                if any(word in task2.passage.lower() for word in task1.passage.lower().split() if len(word) > 5):
                    coherent = True
                    print(f"✅ Tasks {i+1} and {j+1} share concepts (COHERENT)")
    
    if not coherent:
        print("❌ Tasks are INDEPENDENT - no shared concepts or building blocks")
        print("   Each task is self-contained and doesn't reference others")
        print("   This means curriculum structure provides NO benefit!")
    
    return not coherent

def analyze_reward_function():
    """Analyze the reward function to see if it properly rewards curriculum."""
    print("\n" + "="*70)
    print("REWARD FUNCTION ANALYSIS")
    print("="*70)
    
    print("\nReward function components:")
    print("  - Base reward: accuracy_after - accuracy_before (improvement)")
    print("  - Difficulty bonus: Harder tasks get bigger bonus")
    print("    * trivial: 0.2, easy: 0.5, medium: 1.0, hard: 2.0")
    print("    * expert: 3.0, master: 4.0, grandmaster: 5.0")
    print("  - Review bonus: +1.0 if review and improvement > 0")
    print("  - Review penalty: -0.5 if review and accuracy > 0.9")
    
    print("\n" + "="*70)
    print("POTENTIAL ISSUES:")
    print("="*70)
    
    print("\n1. Difficulty bonus might be too large:")
    print("   - Grandmaster task gets +5.0 bonus even if student doesn't learn")
    print("   - Teacher might select hard tasks just for bonus, not learning")
    
    print("\n2. No curriculum coherence reward:")
    print("   - Reward doesn't consider if tasks build on each other")
    print("   - Teacher doesn't get extra reward for coherent sequences")
    
    print("\n3. Reward based on immediate improvement only:")
    print("   - Doesn't consider long-term learning effects")
    print("   - Random might get lucky with good sequences")

def analyze_teacher_behavior():
    """Analyze teacher action selection."""
    print("\n" + "="*70)
    print("TEACHER BEHAVIOR ANALYSIS")
    print("="*70)
    
    generator = MockTaskGenerator(seed=42)
    teacher = TeacherAgent(exploration_bonus=2.0, task_generator=generator)
    student = MockStudentAgent(seed=42)
    
    print(f"\nAction space: {teacher.num_actions} actions")
    print(f"Exploration bonus: {teacher.exploration_bonus}")
    
    # Simulate first 20 selections
    print("\nFirst 20 teacher selections:")
    print("-" * 70)
    
    difficulty_counts = {}
    topic_counts = {}
    
    for i in range(20):
        state = student.get_state()
        action = teacher.select_action(state)
        
        # Simulate reward (random for demo)
        reward = compute_reward(0.5, 0.6, action.difficulty, action.is_review)
        teacher.update(action, reward)
        
        difficulty_counts[action.difficulty] = difficulty_counts.get(action.difficulty, 0) + 1
        topic_counts[action.topic] = topic_counts.get(action.topic, 0) + 1
        
        if i < 10:  # Show first 10
            print(f"{i+1:2d}. {action.topic:15s} | {action.difficulty:12s} | Review: {action.is_review}")
    
    print("\n" + "="*70)
    print("ANALYSIS:")
    print("="*70)
    print("\nDifficulty distribution:")
    for diff, count in sorted(difficulty_counts.items()):
        print(f"  {diff:12s}: {count:3d} times")
    
    print("\nTopic distribution (first 20):")
    for topic, count in sorted(topic_counts.items(), key=lambda x: -x[1])[:5]:
        print(f"  {topic:15s}: {count:3d} times")
    
    print("\n⚠️  With 210 actions and exploration_bonus=2.0:")
    print("   - Teacher might be exploring too much")
    print("   - Takes ~210 iterations just to try each action once")
    print("   - Random strategy gets immediate good mix")

def main():
    print("="*70)
    print("DIAGNOSING RANDOM VS TEACHER PERFORMANCE")
    print("="*70)
    
    # Analyze task coherence
    tasks_independent = analyze_task_coherence()
    
    # Analyze reward function
    analyze_reward_function()
    
    # Analyze teacher behavior
    analyze_teacher_behavior()
    
    # Conclusions
    print("\n" + "="*70)
    print("CONCLUSIONS & RECOMMENDATIONS")
    print("="*70)
    
    if tasks_independent:
        print("\n✅ CONFIRMED: Tasks are INDEPENDENT (incoherent)")
        print("   - Each task is self-contained")
        print("   - No building blocks or prerequisite concepts")
        print("   - Curriculum structure provides minimal benefit")
    
    print("\n🔧 RECOMMENDED FIXES:")
    print("\n1. Make tasks coherent:")
    print("   - Add prerequisite concepts")
    print("   - Reference previous tasks")
    print("   - Build complexity gradually within topics")
    
    print("\n2. Adjust reward function:")
    print("   - Reduce difficulty bonus (currently too high)")
    print("   - Add curriculum coherence bonus")
    print("   - Consider long-term learning effects")
    
    print("\n3. Adjust teacher exploration:")
    print("   - Reduce exploration_bonus (currently 2.0)")
    print("   - Or use Thompson Sampling instead of UCB")
    print("   - Consider action space reduction")

if __name__ == "__main__":
    main()

