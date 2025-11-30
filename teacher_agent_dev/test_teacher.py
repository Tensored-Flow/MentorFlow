"""Unit tests for Teacher Agent system."""

import sys
from pathlib import Path
import importlib

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from mock_student import MockStudentAgent
from mock_task_generator import MockTaskGenerator
from teacher_agent import TeacherAgent
from interfaces import TeacherAction


def test_mock_student_learning():
    """Test that mock student learns."""
    print("Testing student learning...", end=" ")
    
    student = MockStudentAgent(learning_rate=0.15, forgetting_rate=0.05)
    generator = MockTaskGenerator()
    
    # Test learning
    topic = 'history'
    tasks = [generator.generate_task(topic, 'easy') for _ in range(20)]
    
    accuracies = []
    for task in tasks:
        eval_tasks = [generator.generate_task(topic, 'easy') for _ in range(10)]
        acc = student.evaluate(eval_tasks)
        accuracies.append(acc)
        student.learn(task)
    
    # Student should improve
    improvement = accuracies[-1] - accuracies[0]
    assert improvement > 0.1, f"Student should improve! Improvement: {improvement:.3f}"
    
    print("✅ PASSED")
    print(f"   Initial accuracy: {accuracies[0]:.3f}")
    print(f"   Final accuracy: {accuracies[-1]:.3f}")
    print(f"   Improvement: {improvement:.3f}")


def test_mock_student_forgetting():
    """Test that mock student forgets over time."""
    print("Testing student forgetting...", end=" ")
    
    student = MockStudentAgent(learning_rate=0.15, forgetting_rate=0.1)
    generator = MockTaskGenerator()
    
    # Train on one topic
    topic = 'science'
    for _ in range(30):
        task = generator.generate_task(topic, 'easy')
        student.learn(task)
    
    # Measure accuracy
    eval_tasks = [generator.generate_task(topic, 'easy') for _ in range(10)]
    acc_before = student.evaluate(eval_tasks)
    
    # Time passes without practice
    student.advance_time(50.0)
    
    acc_after = student.evaluate(eval_tasks)
    
    # Student should forget
    assert acc_after < acc_before - 0.05, f"Student should forget! Before: {acc_before:.3f}, After: {acc_after:.3f}"
    
    print("✅ PASSED")
    print(f"   Accuracy before forgetting: {acc_before:.3f}")
    print(f"   Accuracy after 50 time units: {acc_after:.3f}")
    print(f"   Forgetting: {acc_before - acc_after:.3f}")


def test_mock_student_initial_accuracy():
    """Test that student starts at ~25% accuracy (random guessing)."""
    print("Testing initial student accuracy...", end=" ")
    
    student = MockStudentAgent()
    generator = MockTaskGenerator()
    
    # Evaluate on many tasks
    eval_tasks = [generator.generate_task('history', 'easy') for _ in range(100)]
    initial_acc = student.evaluate(eval_tasks)
    
    # Should be around 25% (random guessing on 4-choice MCQ)
    assert 0.15 < initial_acc < 0.35, f"Initial accuracy should be ~25%! Got: {initial_acc:.3f}"
    
    print("✅ PASSED")
    print(f"   Initial accuracy: {initial_acc:.3f} (~25% expected)")


def test_teacher_exploration():
    """Test that teacher explores all actions."""
    print("Testing teacher exploration...", end=" ")
    
    teacher = TeacherAgent(exploration_bonus=5.0)  # High exploration
    from mock_student import MockStudentAgent
    from interfaces import StudentState
    
    # Create minimal student state
    student = MockStudentAgent()
    
    actions_tried = set()
    for _ in range(100):
        student_state = student.get_state()
        action = teacher.select_action(student_state)
        actions_tried.add((action.topic, action.difficulty, action.is_review))
        teacher.update(action, 0.0)  # Neutral reward
    
    # Teacher should explore many actions (now has 15 topics × 7 difficulties × 2 = 210 actions)
    expected_actions = 15 * 7 * 2  # topics × difficulties × review options
    assert len(actions_tried) > 20, f"Teacher should explore many actions! Only tried: {len(actions_tried)}"
    
    print("✅ PASSED")
    print(f"   Unique actions tried: {len(actions_tried)}/{expected_actions}")


def test_teacher_exploitation():
    """Test that teacher exploits good actions."""
    print("Testing teacher exploitation...", end=" ")
    
    teacher = TeacherAgent(exploration_bonus=0.1)  # Very low exploration
    from mock_student import MockStudentAgent
    
    student = MockStudentAgent()
    
    # Manually set one action to be very good
    best_action = TeacherAction(topic='history', difficulty='easy', is_review=False)
    best_action_idx = teacher._action_to_index(best_action)
    
    # First, try all actions once (cold start)
    for i in range(teacher.num_actions):
        test_action = teacher._index_to_action(i)
        if i == best_action_idx:
            teacher.update(test_action, 100.0)  # Very high reward
        else:
            teacher.update(test_action, 0.0)  # Low reward
    
    # Now teacher should prefer the best action
    selections = []
    for _ in range(50):  # More samples for better statistics
        student_state = student.get_state()
        action = teacher.select_action(student_state)
        idx = teacher._action_to_index(action)
        selections.append(idx == best_action_idx)
    
    # Should select best action frequently
    exploit_rate = sum(selections) / len(selections)
    assert exploit_rate > 0.3, f"Teacher should exploit good actions! Exploit rate: {exploit_rate:.2f}"
    
    print("✅ PASSED")
    print(f"   Best action selection rate: {exploit_rate:.2f}")


def test_teacher_action_encoding():
    """Test that action encoding/decoding works correctly."""
    print("Testing action encoding/decoding...", end=" ")
    
    teacher = TeacherAgent()
    
    # Test all actions
    for idx in range(teacher.num_actions):
        action1 = teacher._index_to_action(idx)
        idx2 = teacher._action_to_index(action1)
        action2 = teacher._index_to_action(idx2)
        
        assert idx == idx2, f"Encoding mismatch! {idx} != {idx2}"
        assert action1.topic == action2.topic, "Topic mismatch"
        assert action1.difficulty == action2.difficulty, "Difficulty mismatch"
        assert action1.is_review == action2.is_review, "Review flag mismatch"
    
    print("✅ PASSED")
    print(f"   Tested {teacher.num_actions} actions")


def test_task_generator():
    """Test that task generator creates valid tasks."""
    print("Testing task generator...", end=" ")
    
    generator = MockTaskGenerator()
    
    topics = generator.get_available_topics()
    difficulties = generator.get_available_difficulties()
    
    # Check that we have topics and difficulties (exact count may vary after expansion)
    assert len(topics) >= 5, f"Should have at least 5 topics, got {len(topics)}"
    assert len(difficulties) >= 3, f"Should have at least 3 difficulties, got {len(difficulties)}"
    
    # Generate tasks for all combinations
    for topic in topics:
        for difficulty in difficulties:
            task = generator.generate_task(topic, difficulty)
            assert len(task.choices) == 4, "Should have 4 choices"
            assert 0 <= task.answer < 4, "Answer should be valid index"
            assert task.topic == topic, "Topic should match"
            assert task.difficulty == difficulty, "Difficulty should match"
    
    print("✅ PASSED")
    print(f"   Generated tasks for {len(topics)} topics × {len(difficulties)} difficulties")


def test_compare_strategies_uses_ppo_student():
    """Ensure compare_strategies uses PPO student wrapper (not LM student)."""
    print("Testing compare_strategies PPO student selection...", end=" ")
    
    compare_strategies = importlib.import_module("compare_strategies")
    
    assert getattr(compare_strategies, "USE_PPO_STUDENT", False), (
        "PPOStudentWrapper import failed; install stable-baselines3/torch to run comparisons"
    )
    assert hasattr(compare_strategies, "PPOStudentWrapper"), "PPOStudentWrapper missing"
    
    print("✅ PASSED")


def run_all_tests():
    """Run all tests."""
    print("=" * 70)
    print("RUNNING TESTS")
    print("=" * 70)
    print()
    
    tests = [
        test_task_generator,
        test_mock_student_initial_accuracy,
        test_mock_student_learning,
        test_mock_student_forgetting,
        test_teacher_action_encoding,
        test_teacher_exploration,
        test_teacher_exploitation,
        test_compare_strategies_uses_ppo_student,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"❌ FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
        print()
    
    print("=" * 70)
    print(f"TESTS COMPLETE: {passed} passed, {failed} failed")
    print("=" * 70)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
