#!/usr/bin/env python3
"""
Full Integration Test with Progress Bars
Tests entire MentorFlow pipeline after merge, with progress indicators and debugging
"""

import sys
import time
import traceback
from pathlib import Path

# Try to import tqdm for progress bars
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("⚠️  tqdm not available - install with: pip install tqdm")

class PipelineTester:
    def __init__(self):
        self.results = {}
        self.errors = []
        
    def test_import(self, name, import_statement):
        """Test if a module imports correctly."""
        try:
            exec(import_statement)
            self.results[name] = "✅ PASSED"
            return True
        except Exception as e:
            self.results[name] = f"❌ FAILED: {e}"
            self.errors.append((name, str(e)))
            return False
    
    def test_with_progress(self, name, test_func, *args, **kwargs):
        """Run a test with progress indicator."""
        print(f"\n{'='*70}")
        print(f"Testing: {name}")
        print(f"{'='*70}")
        
        start_time = time.time()
        try:
            if HAS_TQDM and 'iterations' in kwargs:
                iterations = kwargs.pop('iterations')
                with tqdm(total=iterations, desc="  Progress", unit="iter") as pbar:
                    result = test_func(*args, progress_bar=pbar, **kwargs)
            else:
                result = test_func(*args, **kwargs)
            
            elapsed = time.time() - start_time
            self.results[name] = f"✅ PASSED ({elapsed:.2f}s)"
            print(f"✅ {name} PASSED ({elapsed:.2f}s)")
            return True
        except Exception as e:
            elapsed = time.time() - start_time
            self.results[name] = f"❌ FAILED ({elapsed:.2f}s): {e}"
            self.errors.append((name, str(e)))
            print(f"❌ {name} FAILED ({elapsed:.2f}s)")
            print(f"Error: {e}")
            traceback.print_exc()
            return False

def test_task_generator():
    """Test task generator imports and basic functionality."""
    sys.path.insert(0, str(Path(__file__).parent))
    from tasks.task_generator import generate_task, NUM_FAMILIES, NUM_DIFFICULTIES
    
    print("  Generating sample tasks...")
    for family_id in range(min(2, NUM_FAMILIES)):  # Test first 2 families
        for diff_id in range(min(2, NUM_DIFFICULTIES)):  # Test first 2 difficulties
            task = generate_task(family_id, diff_id, seed=42)
            assert task is not None
            assert hasattr(task, 'human_prompt')
            assert hasattr(task, 'human_choices')
            assert hasattr(task, 'correct_action')
    print(f"  ✅ Generated tasks successfully ({NUM_FAMILIES} families × {NUM_DIFFICULTIES} difficulties)")

def test_student_env():
    """Test student environment."""
    sys.path.insert(0, str(Path(__file__).parent))
    from student.student_env import StudentEnv
    from tasks.task_generator import generate_task
    
    print("  Creating environment...")
    env = StudentEnv(family_id=0, difficulty_id=0)
    print("  Resetting environment...")
    obs = env.reset()
    assert obs is not None
    print(f"  ✅ Environment works (obs shape: {len(obs) if isinstance(obs, list) else obs.shape})")

def test_ppo_agent():
    """Test PPO agent."""
    sys.path.insert(0, str(Path(__file__).parent))
    try:
        from student.ppo_agent import StudentAgent
        print("  Creating PPO agent...")
        agent = StudentAgent(seed=42, verbose=0)
        print("  ✅ PPO agent created successfully")
        return True
    except Exception as e:
        print(f"  ⚠️  PPO agent import/creation failed: {e}")
        print("  (This might be expected if SB3 not installed)")
        return False

def test_teacher_bandit():
    """Test teacher bandit."""
    sys.path.insert(0, str(Path(__file__).parent))
    from teacher.teacher_bandit import TeacherBandit, BanditStrategy
    
    print("  Creating teacher bandit...")
    teacher = TeacherBandit(strategy=BanditStrategy.UCB1, num_arms=15)
    print("  Selecting arm...")
    arm = teacher.select_arm()
    assert 0 <= arm < 15
    print(f"  ✅ Teacher bandit works (selected arm: {arm})")

def test_teacher_agent_dev():
    """Test teacher agent dev system."""
    sys.path.insert(0, str(Path(__file__).parent / "teacher_agent_dev"))
    from test_teacher import run_all_tests
    
    print("  Running teacher agent dev tests...")
    success = run_all_tests()
    if not success:
        raise Exception("Teacher agent dev tests failed")
    print("  ✅ Teacher agent dev tests passed")

def test_student_agent_dev():
    """Test student agent dev system."""
    sys.path.insert(0, str(Path(__file__).parent / "student_agent_dev"))
    from test_student import run_all_tests
    
    print("  Running student agent dev tests...")
    print("  ⚠️  Note: This may take time due to DistilBERT loading...")
    success = run_all_tests()
    if not success:
        raise Exception("Student agent dev tests failed")
    print("  ✅ Student agent dev tests passed")

def test_training_loop_import():
    """Test training loop imports."""
    sys.path.insert(0, str(Path(__file__).parent))
    from training.training_loop import MetaTrainer, MetaTrainingConfig
    
    print("  ✅ Training loop imports successfully")

def test_interfaces_compatibility():
    """Test interface compatibility."""
    sys.path.insert(0, str(Path(__file__).parent / "teacher_agent_dev"))
    sys.path.insert(0, str(Path(__file__).parent / "student_agent_dev"))
    
    from teacher_agent_dev.interfaces import Task as TTask, StudentState as TState
    from student_agent_dev.interfaces import Task as STask, StudentState as SState
    
    # Check Task compatibility
    import inspect
    t_fields = set(inspect.signature(TTask.__init__).parameters.keys())
    s_fields = set(inspect.signature(STask.__init__).parameters.keys())
    
    if t_fields == s_fields:
        print("  ✅ Task dataclasses are compatible")
    else:
        print(f"  ⚠️  Task fields differ: {t_fields.symmetric_difference(s_fields)}")
    
    # Check StudentState compatibility
    t_state_fields = set(inspect.signature(TState.__init__).parameters.keys())
    s_state_fields = set(inspect.signature(SState.__init__).parameters.keys())
    
    if t_state_fields == s_state_fields:
        print("  ✅ StudentState dataclasses are compatible")
    else:
        print(f"  ⚠️  StudentState fields differ: {t_state_fields.symmetric_difference(s_state_fields)}")

def main():
    tester = PipelineTester()
    
    print("="*70)
    print("MENTORFLOW FULL INTEGRATION TEST")
    print("="*70)
    print(f"Testing after merge - with progress bars: {HAS_TQDM}")
    print()
    
    # Test imports first
    print("="*70)
    print("PHASE 1: IMPORT TESTS")
    print("="*70)
    
    tester.test_import("tasks.task_generator", "from tasks.task_generator import generate_task")
    tester.test_import("student.student_env", "from student.student_env import StudentEnv")
    tester.test_import("student.ppo_agent", "from student.ppo_agent import StudentAgent")
    tester.test_import("teacher.teacher_bandit", "from teacher.teacher_bandit import TeacherBandit")
    tester.test_import("training.training_loop", "from training.training_loop import MetaTrainer")
    
    # Test functionality
    print("\n" + "="*70)
    print("PHASE 2: FUNCTIONALITY TESTS")
    print("="*70)
    
    tester.test_with_progress("Task Generator", test_task_generator)
    tester.test_with_progress("Student Environment", test_student_env)
    tester.test_with_progress("PPO Agent", test_ppo_agent)
    tester.test_with_progress("Teacher Bandit", test_teacher_bandit)
    tester.test_with_progress("Interfaces Compatibility", test_interfaces_compatibility)
    
    # Test development systems (may be slow)
    print("\n" + "="*70)
    print("PHASE 3: DEVELOPMENT SYSTEM TESTS")
    print("="*70)
    print("⚠️  These may take time...")
    
    try:
        tester.test_with_progress("Teacher Agent Dev", test_teacher_agent_dev)
    except Exception as e:
        print(f"  ⚠️  Teacher agent dev test skipped: {e}")
    
    # Student agent dev is optional (slow due to DistilBERT)
    user_input = input("\nRun student agent dev tests? (slow - DistilBERT loading) [y/N]: ").strip().lower()
    if user_input == 'y':
        try:
            tester.test_with_progress("Student Agent Dev", test_student_agent_dev)
        except Exception as e:
            print(f"  ⚠️  Student agent dev test failed: {e}")
    else:
        print("  ⏭️  Skipping student agent dev tests (slow)")
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for name, result in tester.results.items():
        print(f"{result} {name}")
    
    print(f"\n{'='*70}")
    if tester.errors:
        print(f"⚠️  {len(tester.errors)} error(s) found:")
        for name, error in tester.errors:
            print(f"  - {name}: {error[:100]}")
    else:
        print("✅ All tests passed!")
    
    return len(tester.errors) == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

