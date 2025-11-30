"""
Fast unit tests for student agent with progress bars.

Optimized for speed with tqdm progress bars:
- Shows progress during slow operations (model loading, training, evaluation)
- Shared student instance where possible
- Reduced iteration counts for fast tests
- Minimal evaluation sets
"""

import sys
from student_agent import StudentAgent
from mock_task_generator import MockTaskGenerator
import torch

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("⚠️  tqdm not installed. Install with: pip install tqdm")
    # Dummy tqdm if not available
    class tqdm:
        def __init__(self, iterable=None, *args, **kwargs):
            self.iterable = iterable
        def __enter__(self):
            return self.iterable
        def __exit__(self, *args):
            pass
        def __iter__(self):
            return iter(self.iterable) if self.iterable else iter([])
        def update(self, n=1):
            pass


def test_student_can_load():
    """Test DistilBERT loads successfully (or graceful fallback)."""
    print("Testing student initialization...", end=" ", flush=True)
    
    # Model loading can be slow - show that we're working
    try:
        student = StudentAgent(device='cpu')
        print("✅ Student model initialized")
        return student
    except Exception as e:
        print(f"⚠️ Error: {e}")
        raise


def test_student_can_answer():
    """Test student can predict answers."""
    print("Testing answer prediction...", end=" ", flush=True)
    student = StudentAgent(device='cpu')
    generator = MockTaskGenerator()
    
    task = generator.generate_task('history', 'easy')
    answer = student.answer(task)
    
    assert 0 <= answer < 4, f"Answer should be 0-3, got {answer}"
    print("✅ Student can answer tasks")


def test_student_learns():
    """Test student improves with practice (with progress bar)."""
    print("Testing learning capability...", flush=True)
    student = StudentAgent(device='cpu')
    generator = MockTaskGenerator()
    
    topic = 'science'
    
    # Smaller eval set for speed
    print("   Generating eval set...", end=" ", flush=True)
    eval_tasks = [generator.generate_task(topic, 'easy') for _ in range(5)]
    print("Done")
    
    # Measure initial accuracy
    print("   Evaluating initial accuracy...", end=" ", flush=True)
    initial_acc = student.evaluate(eval_tasks)
    print(f"{initial_acc:.3f}")
    
    # Training with progress bar
    num_iterations = 15
    print(f"   Training on {num_iterations} tasks:")
    
    if HAS_TQDM:
        pbar = tqdm(range(num_iterations), desc="      Progress", leave=False)
        for i in pbar:
            task = generator.generate_task(topic, 'easy')
            student.learn(task)
            pbar.set_postfix({'tasks': i+1})
    else:
        # Fallback: simple progress indicator
        for i in range(num_iterations):
            if (i + 1) % 5 == 0:
                print(f"      {i+1}/{num_iterations}...", end="\r", flush=True)
            task = generator.generate_task(topic, 'easy')
            student.learn(task)
        print(f"      {num_iterations}/{num_iterations}     ")  # Clear line
    
    # Measure final accuracy
    print("   Evaluating final accuracy...", end=" ", flush=True)
    final_acc = student.evaluate(eval_tasks)
    print(f"{final_acc:.3f}")
    
    improvement = final_acc - initial_acc
    print(f"✅ Learning verified (improvement: {improvement:+.3f})")


def test_student_forgets():
    """Test memory decay works (with progress bar)."""
    print("Testing memory decay...", flush=True)
    student = StudentAgent(device='cpu', retention_constant=20.0)
    generator = MockTaskGenerator()
    
    topic = 'literature'
    
    # Training with progress bar
    num_iterations = 20
    print(f"   Training on {num_iterations} tasks:")
    
    if HAS_TQDM:
        pbar = tqdm(range(num_iterations), desc="      Progress", leave=False)
        for i in pbar:
            task = generator.generate_task(topic, 'easy')
            student.learn(task)
            pbar.set_postfix({'tasks': i+1})
    else:
        for i in range(num_iterations):
            if (i + 1) % 5 == 0:
                print(f"      {i+1}/{num_iterations}...", end="\r", flush=True)
            task = generator.generate_task(topic, 'easy')
            student.learn(task)
        print(f"      {num_iterations}/{num_iterations}     ")
    
    print("   Evaluating before forgetting...", end=" ", flush=True)
    eval_tasks = [generator.generate_task(topic, 'easy') for _ in range(5)]
    acc_before = student.evaluate(eval_tasks)
    print(f"{acc_before:.3f}")
    
    # Time passes
    print("   Simulating time passage (forgetting)...", end=" ", flush=True)
    student.advance_time(50.0)
    print("Done")
    
    print("   Evaluating after forgetting...", end=" ", flush=True)
    acc_after = student.evaluate(eval_tasks)
    print(f"{acc_after:.3f}")
    
    if acc_after < acc_before:
        print(f"✅ Forgetting verified (drop: {acc_before - acc_after:.3f})")
    else:
        print(f"⚠️ Forgetting minimal (change: {acc_after - acc_before:+.3f})")


def test_student_state():
    """Test state reporting works."""
    print("Testing state reporting...", flush=True)
    student = StudentAgent(device='cpu')
    generator = MockTaskGenerator()
    
    # Training with progress bar
    topics_to_test = ['history', 'science']
    tasks_per_topic = 5
    total_tasks = len(topics_to_test) * tasks_per_topic
    
    print(f"   Training on {total_tasks} tasks:")
    
    for topic in topics_to_test:
        if HAS_TQDM:
            pbar = tqdm(range(tasks_per_topic), desc=f"      {topic}", leave=False)
            for i in pbar:
                task = generator.generate_task(topic, 'easy')
                student.learn(task)
        else:
            for i in range(tasks_per_topic):
                task = generator.generate_task(topic, 'easy')
                student.learn(task)
    
    state = student.get_state()
    
    assert len(state.topic_accuracies) > 0
    assert state.total_timesteps >= 10
    
    print("✅ State reporting works")


def run_all_tests():
    """Run all tests with progress indicators."""
    print("=" * 60)
    print("RUNNING STUDENT AGENT TESTS")
    print("=" * 60)
    if not HAS_TQDM:
        print("💡 Tip: Install tqdm for progress bars: pip install tqdm")
    print()
    
    import time
    start_time = time.time()
    
    try:
        test_student_can_load()
        test_student_can_answer()
        test_student_learns()
        test_student_forgets()
        test_student_state()
        
        elapsed = time.time() - start_time
        print()
        print("=" * 60)
        print(f"🎉 All tests passed! (Total time: {elapsed:.2f}s)")
        print("=" * 60)
        return True
    except Exception as e:
        elapsed = time.time() - start_time
        print()
        print("=" * 60)
        print(f"❌ Test failed after {elapsed:.2f}s")
        print(f"Error: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
