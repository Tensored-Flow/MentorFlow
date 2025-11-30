"""Diagnose why training might be slow."""
import sys
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).parent))

print("=" * 70, flush=True)
print("DIAGNOSING TRAINING SPEED", flush=True)
print("=" * 70, flush=True)

# 1. Test environment step speed
print("\n1. Testing environment step speed...", flush=True)
from student.student_env import StudentEnv

env = StudentEnv(family_id=0, difficulty_id=0, seed=42)
obs, info = env.reset()

start = time.time()
episode_count = 0
total_steps = 0

for _ in range(100):  # Run 100 episodes
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        total_steps += 1
    obs, info = env.reset()
    episode_count += 1

elapsed = time.time() - start
print(f"  Completed {episode_count} episodes ({total_steps} steps) in {elapsed:.2f}s", flush=True)
print(f"  Speed: ~{total_steps/elapsed:.0f} steps/sec, ~{episode_count/elapsed:.0f} episodes/sec", flush=True)

# 2. Calculate rollout collection time
print("\n2. Calculating rollout collection requirements...", flush=True)
n_steps = 256
avg_episode_length = total_steps / episode_count
episodes_per_rollout = n_steps / avg_episode_length
time_per_rollout = episodes_per_rollout * (elapsed / episode_count)

print(f"  With n_steps={n_steps}:", flush=True)
print(f"    Average episode length: ~{avg_episode_length:.1f} steps", flush=True)
print(f"    Episodes per rollout: ~{episodes_per_rollout:.1f}", flush=True)
print(f"    Estimated time per rollout: ~{time_per_rollout:.2f}s", flush=True)

# 3. Check total_timesteps requirement
print("\n3. Training time estimates...", flush=True)
for total_timesteps in [100, 500, 1000, 5000, 50000]:
    rollouts_needed = max(1, total_timesteps // n_steps)
    estimated_time = rollouts_needed * time_per_rollout
    print(f"  total_timesteps={total_timesteps:,}:", flush=True)
    print(f"    ~{rollouts_needed} rollouts needed", flush=True)
    print(f"    Estimated time: ~{estimated_time:.1f}s ({estimated_time/60:.1f} min)", flush=True)

print("\n" + "=" * 70, flush=True)
print("DIAGNOSIS COMPLETE", flush=True)
print("=" * 70, flush=True)
print("\nRecommendations:", flush=True)
print("  - Progress bar only updates after each rollout completes", flush=True)
print("  - For faster feedback, use smaller n_steps (e.g., 64 or 128)", flush=True)
print("  - For quick tests, use total_timesteps >= 1000 to see progress", flush=True)
print("  - Real training should use total_timesteps >= 5000", flush=True)

