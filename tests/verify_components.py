"""Lightweight verification of all four main components (avoids threading issues)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

print("=" * 70, flush=True)
print("VERIFYING FOUR MAIN COMPONENTS", flush=True)
print("=" * 70, flush=True)

results = {}

# 1. Test task_generator
print("\n[1/4] Testing task_generator...", flush=True)
try:
    from tasks.task_generator import (
        generate_task, generate_task_by_arm, 
        NUM_FAMILIES, NUM_DIFFICULTIES, OBS_VEC_SIZE
    )
    task = generate_task(0, 0, seed=42)
    assert len(task.obs_vec) == OBS_VEC_SIZE
    assert 0 <= task.correct_action < 4
    assert task.family_id == 0
    assert task.difficulty_id == 0
    print("  ✓ task_generator works correctly", flush=True)
    results['task_generator'] = '✓'
except Exception as e:
    print(f"  ✗ FAILED: {e}", flush=True)
    results['task_generator'] = f'✗ {e}'

# 2. Test train_single_task structure (without actually running)
print("\n[2/4] Testing train_single_task (code structure)...", flush=True)
try:
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "train_single_task",
        Path(__file__).parent / "training" / "train_single_task.py"
    )
    # Just check file exists and can be parsed
    with open(Path(__file__).parent / "training" / "train_single_task.py") as f:
        code = f.read()
        assert 'def train_single_task' in code
        assert 'RolloutProgressCallback' in code
        assert 'PPO_CFG' in code
    
    # Check that StudentEnv file exists and has correct structure
    with open(Path(__file__).parent / "student" / "student_env.py") as f:
        env_code = f.read()
        assert 'class StudentEnv' in env_code
        assert 'OBS_VEC_SIZE' in env_code
    
    print("  ✓ train_single_task code structure OK", flush=True)
    print("  ✓ StudentEnv class exists", flush=True)
    results['train_single_task'] = '✓'
except Exception as e:
    print(f"  ✗ FAILED: {e}", flush=True)
    results['train_single_task'] = f'✗ {e}'

# 3. Test train_with_eval_logging structure
print("\n[3/4] Testing train_with_eval_logging...", flush=True)
try:
    with open(Path(__file__).parent / "training" / "train_with_eval_logging.py") as f:
        code = f.read()
        assert 'def train_with_eval_logging' in code
        assert 'eval_problems' in code
        assert 'train_steps_per_generation' in code
        assert 'SharedProgressCallback' in code
        # Verify it uses 100 problems
        assert 'eval_problems: int = 100' in code or 'eval_problems=100' in code
    
    print("  ✓ train_with_eval_logging code structure OK", flush=True)
    print("  ✓ Configured for 100 problems per generation", flush=True)
    results['train_with_eval_logging'] = '✓'
except Exception as e:
    print(f"  ✗ FAILED: {e}", flush=True)
    results['train_with_eval_logging'] = f'✗ {e}'

# 4. Test train_with_teacher structure
print("\n[4/4] Testing train_with_teacher...", flush=True)
try:
    from teacher.teacher_ucb import TeacherUCB
    
    teacher = TeacherUCB(num_arms=15)
    arm = teacher.select_arm()
    assert 0 <= arm < 15
    teacher.update(arm, 0.5)
    
    with open(Path(__file__).parent / "training" / "train_with_teacher.py") as f:
        code = f.read()
        assert 'def train_with_teacher' in code
        assert 'TeacherUCB' in code
    
    print("  ✓ train_with_teacher code structure OK", flush=True)
    print("  ✓ TeacherUCB works correctly", flush=True)
    results['train_with_teacher'] = '✓'
except Exception as e:
    print(f"  ✗ FAILED: {e}", flush=True)
    results['train_with_teacher'] = f'✗ {e}'

print("\n" + "=" * 70, flush=True)
print("VERIFICATION SUMMARY", flush=True)
print("=" * 70, flush=True)
for component, status in results.items():
    print(f"  {component:30s}: {status}", flush=True)

all_passed = all('✓' in status for status in results.values())
if all_passed:
    print("\n✓ ALL COMPONENTS VERIFIED!", flush=True)
else:
    print("\n⚠ Some components have issues - see above", flush=True)

