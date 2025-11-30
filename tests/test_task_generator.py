"""Fast minimal test for task generator - shows actual questions."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from tasks.task_generator import (
    generate_task, 
    NUM_FAMILIES, 
    NUM_DIFFICULTIES, 
    OBS_VEC_SIZE, 
    NUM_CHOICES,
    FAMILY_NAMES,
    DIFFICULTY_NAMES
)

print("=" * 70, flush=True)
print("TASK GENERATOR - SHOWING ACTUAL QUESTIONS", flush=True)
print("=" * 70, flush=True)

# Generate and display tasks from each family
print("\n" + "=" * 70, flush=True)
print("SAMPLE QUESTIONS FROM EACH FAMILY", flush=True)
print("=" * 70, flush=True)

for family_id in range(NUM_FAMILIES):
    for difficulty_id in range(NUM_DIFFICULTIES):
        task = generate_task(family_id, difficulty_id, seed=42 + family_id * 10 + difficulty_id)
        
        assert len(task.obs_vec) == OBS_VEC_SIZE
        assert 0 <= task.correct_action < NUM_CHOICES
        
        print(f"\n[Family: {FAMILY_NAMES[family_id]}, Difficulty: {DIFFICULTY_NAMES[difficulty_id]}]", flush=True)
        print("-" * 70, flush=True)
        print(task.human_prompt, flush=True)
        print("\nChoices:", flush=True)
        for i, choice in enumerate(task.human_choices):
            marker = "✓" if i == task.correct_action else " "
            print(f"  {marker} [{i}] {choice}", flush=True)
        print("-" * 70, flush=True)

print("\n" + "=" * 70, flush=True)
print("VERIFICATION TESTS", flush=True)
print("=" * 70, flush=True)

# Quick verification
print("\nTesting determinism...", flush=True)
t1 = generate_task(0, 0, seed=123)
t2 = generate_task(0, 0, seed=123)
assert t1.obs_vec == t2.obs_vec
assert t1.correct_action == t2.correct_action
print("  ✓ Determinism OK (same seed = same task)", flush=True)

print("\n✓ Task generator works correctly!", flush=True)
print("=" * 70, flush=True)

