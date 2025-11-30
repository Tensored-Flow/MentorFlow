"""Verify that Teacher Agent is actually learning and improving."""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from train_teacher import train_teacher
from teacher_agent import TeacherAgent
from interfaces import StudentState


def verify_teacher_improves():
    """Verify teacher agent's reward increases over time."""
    print("=" * 70)
    print("VERIFYING TEACHER AGENT LEARNING")
    print("=" * 70)
    
    # Train teacher
    print("\nTraining teacher for 500 iterations...")
    history, teacher, student = train_teacher(num_iterations=500, verbose=False)
    
    # Analyze rewards over time
    rewards = np.array(history['teacher_rewards'])
    
    # Split into early and late phases
    early_rewards = rewards[:100]
    mid_rewards = rewards[100:300]
    late_rewards = rewards[300:]
    
    early_avg = np.mean(early_rewards)
    mid_avg = np.mean(mid_rewards)
    late_avg = np.mean(late_rewards)
    
    print(f"\nReward Analysis:")
    print(f"  Early (iter 0-99):    {early_avg:.3f}")
    print(f"  Mid (iter 100-299):   {mid_avg:.3f}")
    print(f"  Late (iter 300-499):  {late_avg:.3f}")
    
    # Check if teacher is learning
    improvement = late_avg - early_avg
    print(f"\n  Improvement: {improvement:+.3f}")
    
    if improvement > 0.2:
        print("  ✅ Teacher is learning! (late rewards > early rewards)")
    elif improvement > 0:
        print("  ⚠️ Teacher shows slight improvement")
    else:
        print("  ❌ Teacher is NOT learning (rewards decreasing or flat)")
    
    # Check if teacher is exploiting good actions
    stats = teacher.get_statistics()
    
    # Find best actions (highest average reward)
    avg_rewards_per_action = []
    for idx in range(len(stats['action_counts'])):
        if stats['action_counts'][idx] > 0:
            avg_reward = stats['action_rewards'][idx] / stats['action_counts'][idx]
            count = stats['action_counts'][idx]
            avg_rewards_per_action.append((idx, avg_reward, count))
    
    avg_rewards_per_action.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\nTop 5 Actions by Average Reward:")
    for i, (idx, avg_reward, count) in enumerate(avg_rewards_per_action[:5]):
        action = teacher._index_to_action(idx)
        print(f"  {i+1}. {action.topic}-{action.difficulty}-{'R' if action.is_review else 'N'}: "
              f"avg_reward={avg_reward:.3f}, count={count}")
    
    # Check if teacher preferentially selects high-reward actions in late phase
    print(f"\nAction Selection Analysis (Late Phase):")
    late_actions = history['actions'][300:]
    late_rewards_for_actions = history['teacher_rewards'][300:]
    
    # Group by action
    action_reward_map = {}
    for action, reward in zip(late_actions, late_rewards_for_actions):
        key = (action.topic, action.difficulty, action.is_review)
        if key not in action_reward_map:
            action_reward_map[key] = []
        action_reward_map[key].append(reward)
    
    # Get top actions by frequency in late phase
    action_counts_late = {}
    for action in late_actions:
        key = (action.topic, action.difficulty, action.is_review)
        action_counts_late[key] = action_counts_late.get(key, 0) + 1
    
    sorted_actions = sorted(action_counts_late.items(), key=lambda x: x[1], reverse=True)
    
    print(f"  Most frequently selected actions in late phase:")
    for i, ((topic, diff, review), count) in enumerate(sorted_actions[:5]):
        avg_reward = np.mean(action_reward_map.get((topic, diff, review), [0]))
        print(f"    {i+1}. {topic[:3]}-{diff[:2]}-{'R' if review else 'N'}: "
              f"count={count}, avg_reward={avg_reward:.3f}")
    
    # Verify teacher is using learned information
    print(f"\n" + "=" * 70)
    print("VERIFICATION RESULTS:")
    print("=" * 70)
    
    checks_passed = 0
    total_checks = 4
    
    # Check 1: Rewards improve over time
    if improvement > 0.1:
        print("✅ Check 1: Teacher rewards improve over time")
        checks_passed += 1
    else:
        print("❌ Check 1: Teacher rewards do not improve significantly")
    
    # Check 2: Teacher tries all actions (exploration)
    unique_actions = len([c for c in stats['action_counts'] if c > 0])
    if unique_actions >= 25:
        print(f"✅ Check 2: Teacher explores actions ({unique_actions}/30)")
        checks_passed += 1
    else:
        print(f"❌ Check 2: Teacher doesn't explore enough ({unique_actions}/30)")
    
    # Check 3: Teacher has some preference (exploitation)
    top_action_freq = sorted_actions[0][1] if sorted_actions else 0
    if top_action_freq > 20:
        print(f"✅ Check 3: Teacher shows preference (top action selected {top_action_freq} times)")
        checks_passed += 1
    else:
        print(f"❌ Check 3: Teacher doesn't show strong preference")
    
    # Check 4: Student improves (teacher's goal)
    student_early = np.mean(history['student_accuracies'][:100])
    student_late = np.mean(history['student_accuracies'][300:])
    student_improvement = student_late - student_early
    if student_improvement > 0.1:
        print(f"✅ Check 4: Student improves significantly ({student_early:.3f} → {student_late:.3f})")
        checks_passed += 1
    else:
        print(f"❌ Check 4: Student doesn't improve much")
    
    print(f"\nTotal: {checks_passed}/{total_checks} checks passed")
    
    if checks_passed >= 3:
        print("\n✅ TEACHER AGENT IS LEARNING AND IMPROVING!")
    else:
        print("\n⚠️ Teacher agent may need tuning")
    
    print("=" * 70)
    
    return checks_passed >= 3


def verify_ucb_algorithm():
    """Verify UCB algorithm is working correctly."""
    print("\n" + "=" * 70)
    print("VERIFYING UCB ALGORITHM")
    print("=" * 70)
    
    teacher = TeacherAgent(exploration_bonus=2.0)
    
    # Test: Give some actions high rewards
    from interfaces import TeacherAction
    
    good_action = TeacherAction(topic='history', difficulty='easy', is_review=False)
    bad_action = TeacherAction(topic='science', difficulty='hard', is_review=False)
    
    # Give good action high rewards multiple times
    for _ in range(10):
        teacher.update(good_action, 10.0)
    
    # Give bad action low rewards
    for _ in range(10):
        teacher.update(bad_action, 0.5)
    
    # Teacher should prefer good action
    from mock_student import MockStudentAgent
    
    student = MockStudentAgent()
    selections = []
    
    for _ in range(50):
        student_state = student.get_state()
        action = teacher.select_action(student_state)
        selections.append(action)
    
    good_selections = sum(1 for a in selections if a.topic == 'history' and a.difficulty == 'easy' and not a.is_review)
    good_rate = good_selections / len(selections)
    
    print(f"\nGood action selection rate: {good_rate:.2f}")
    if good_rate > 0.3:
        print("✅ UCB algorithm is working (prefers high-reward actions)")
    else:
        print("❌ UCB algorithm may not be working correctly")
    
    # Verify UCB scores
    ucb_scores = teacher._compute_ucb_scores()
    good_idx = teacher._action_to_index(good_action)
    bad_idx = teacher._action_to_index(bad_action)
    
    print(f"\nUCB Scores:")
    print(f"  Good action (history-easy-N): {ucb_scores[good_idx]:.3f}")
    print(f"  Bad action (science-hard-N):  {ucb_scores[bad_idx]:.3f}")
    
    if ucb_scores[good_idx] > ucb_scores[bad_idx]:
        print("✅ UCB correctly ranks good action higher")
    else:
        print("❌ UCB ranking may be incorrect")
    
    print("=" * 70)


if __name__ == "__main__":
    # Verify UCB algorithm
    verify_ucb_algorithm()
    
    # Verify teacher improves
    print("\n")
    success = verify_teacher_improves()
    
    sys.exit(0 if success else 1)

