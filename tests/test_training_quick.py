"""Quick test script to verify training scripts work."""
import sys
from pathlib import Path
import os

sys.path.insert(0, str(Path(__file__).parent))

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

print("=" * 60, flush=True)
print("QUICK TRAINING TESTS", flush=True)
print("=" * 60, flush=True)

# Test 1: train_single_task
print("\n1. Testing train_single_task (minimal run)...", flush=True)
try:
    from training.train_single_task import train_single_task
    
    # Run with very small timesteps for quick test
    model = train_single_task(
        family_id=0,
        difficulty_id=0,
        total_timesteps=500,  # Very small for quick test
        eval_episodes=50,
        save_dir="models/test"
    )
    print("  ✓ train_single_task completed successfully", flush=True)
except Exception as e:
    print(f"  ✗ train_single_task failed: {e}", flush=True)
    import traceback
    traceback.print_exc()

# Test 2: train_with_eval_logging
print("\n2. Testing train_with_eval_logging (minimal run)...", flush=True)
try:
    from training.train_with_eval_logging import train_with_eval_logging, TrainingConfig
    
    config = TrainingConfig(
        total_timesteps=1000,
        train_steps_per_generation=200,
        eval_problems=50,  # Smaller for quick test
        family_id=0,
        difficulty_id=0,
        save_dir="models/test",
        model_name="test_eval_logged"
    )
    
    model, logger = train_with_eval_logging(config)
    print(f"  ✓ train_with_eval_logging completed successfully", flush=True)
    print(f"    - Evaluations logged: {len(logger.accuracies)}", flush=True)
    if logger.accuracies:
        print(f"    - Final accuracy: {logger.accuracies[-1]:.3f}", flush=True)
except Exception as e:
    print(f"  ✗ train_with_eval_logging failed: {e}", flush=True)
    import traceback
    traceback.print_exc()

# Test 3: train_with_teacher
print("\n3. Testing train_with_teacher (minimal run)...", flush=True)
try:
    from training.train_with_teacher import train_with_teacher
    
    train_with_teacher(
        total_rounds=3,  # Very small for quick test
        train_steps_per_round=200,
        eval_episodes=50,
        eval_task=(0, 0),
        save_path="models/test/ppo_teacher_test.zip"
    )
    print("  ✓ train_with_teacher completed successfully", flush=True)
except Exception as e:
    print(f"  ✗ train_with_teacher failed: {e}", flush=True)
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60, flush=True)
print("TRAINING TESTS COMPLETE", flush=True)
print("=" * 60, flush=True)

