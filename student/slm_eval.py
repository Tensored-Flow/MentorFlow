"""Evaluation script for the SLM-style student.

Generates a broader set of synthetic tasks across topics and difficulty
levels, runs the student for many steps, and prints/plots simple summaries.

Run from repo root:
    python3 -m student.slm_eval

If matplotlib is installed, a PNG plot of average skill vs difficulty
will be saved as 'slm_eval_results.png' in the repo root.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Dict, Any, List, Tuple

from .slm_agent import run_student_step


TOPICS: List[str] = ["history", "science", "literature", "logic", "math"]
DIFFICULTIES: List[int] = list(range(1, 11))  # 1-10


def make_synthetic_task(topic: str, difficulty: int, rng: random.Random) -> Dict[str, Any]:
    """Create a simple synthetic multiple-choice task.

    We don't know the true correct answer; this is just to exercise the
    student's behavior and state dynamics.
    """
    question = f"(Level {difficulty}) A synthetic {topic} question."
    options = {
        "A": "Option A for this question.",
        "B": "Option B for this question.",
        "C": "Option C for this question.",
        "D": "Option D for this question.",
    }
    return {
        "topic": topic,
        "difficulty": difficulty,
        "question": question,
        "options": options,
    }


def run_slm_evaluation(
    num_steps: int = 200,
    seed: int = 0,
) -> Tuple[Dict[Tuple[str, int], float], Dict[Tuple[str, int], int]]:
    """Run the SLM student on many synthetic tasks.

    Returns:
        avg_skills: mapping (topic, difficulty) -> final skill estimate
        counts: how many times each (topic, difficulty) was seen
    """
    rng = random.Random(seed)
    state: Dict[str, Any] = {}
    prev: Dict[str, Any] = {"was_previous_question": False}

    # Track last-seen skills and counts
    skill_sums: Dict[Tuple[str, int], float] = defaultdict(float)
    counts: Dict[Tuple[str, int], int] = defaultdict(int)

    for step in range(1, num_steps + 1):
        topic = rng.choice(TOPICS)
        difficulty = rng.choice(DIFFICULTIES)
        task = make_synthetic_task(topic, difficulty, rng)

        out = run_student_step(
            student_state=state,
            previous_feedback=prev,
            current_task=task,
            rng_seed=step,
        )

        state = out["updated_state"]

        # For this quick eval, pretend correctness is stochastic with
        # probability equal to current skill (clipped to [0,1]).
        skill_val = float(state["skills"][topic][difficulty])
        p_correct = max(0.0, min(1.0, skill_val))
        was_correct = rng.random() < p_correct

        prev = {
            "was_previous_question": True,
            "previous_topic": topic,
            "previous_difficulty": difficulty,
            "previous_correct": was_correct,
        }

        key = (topic, difficulty)
        skill_sums[key] += skill_val
        counts[key] += 1

    avg_skills: Dict[Tuple[str, int], float] = {}
    for key, total in skill_sums.items():
        c = counts[key]
        avg_skills[key] = total / c if c > 0 else 0.0

    return avg_skills, counts


def print_summary(avg_skills: Dict[Tuple[str, int], float], counts: Dict[Tuple[str, int], int]) -> None:
    """Print a simple text summary of skills by difficulty and topic."""
    print("\n=== SLM Evaluation Summary ===")
    for d in DIFFICULTIES:
        # Average across topics
        vals = [avg_skills.get((t, d), 0.0) for t in TOPICS if counts.get((t, d), 0) > 0]
        if not vals:
            continue
        mean_skill = sum(vals) / len(vals)
        print(f"Difficulty {d:2d}: mean skill across topics = {mean_skill:.3f}")


def maybe_plot(avg_skills: Dict[Tuple[str, int], float], counts: Dict[Tuple[str, int], int]) -> None:
    """Optionally plot mean skill vs difficulty using matplotlib if available."""
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        print("matplotlib not installed; skipping plot. Install with 'pip install matplotlib' if desired.")
        return

    xs: List[int] = []
    ys: List[float] = []
    for d in DIFFICULTIES:
        vals = [avg_skills.get((t, d), 0.0) for t in TOPICS if counts.get((t, d), 0) > 0]
        if not vals:
            continue
        xs.append(d)
        ys.append(sum(vals) / len(vals))

    if not xs:
        print("No data to plot.")
        return

    plt.figure(figsize=(6, 4))
    plt.plot(xs, ys, marker="o")
    plt.xlabel("Difficulty level (1 = easy, 10 = hard)")
    plt.ylabel("Mean skill across topics")
    plt.title("SLM Student Skill vs Difficulty")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("slm_eval_results.png")
    print("Saved plot to slm_eval_results.png")


def main() -> None:
    avg_skills, counts = run_slm_evaluation(num_steps=300, seed=0)
    print_summary(avg_skills, counts)
    maybe_plot(avg_skills, counts)


if __name__ == "__main__":
    main()
