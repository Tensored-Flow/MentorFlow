"""Small Language Model style student agent for curriculum-learning experiments.

Implements JSON-based state with skills, forgetting over time, learning from feedback,
and noisy multiple-choice answering.
"""

from __future__ import annotations

import math
import random
from typing import Dict, Any, List


TOPICS: List[str] = ["history", "science", "literature", "logic", "math"]
# Difficulty levels are now an integer scale 1-10, where 1 is easiest and 10 is hardest.
DIFFICULTIES: List[int] = list(range(1, 11))

# Default skills by numeric difficulty: easier levels start higher.
DEFAULT_SKILLS = {d: 0.2 - 0.015 * (d - 1) for d in DIFFICULTIES}  # 0.2 → 0.065
DEFAULT_LAST = {d: 0 for d in DIFFICULTIES}

ALPHA = 0.1
TAU = 20.0


def _ensure_state(student_state: Dict[str, Any], current_step: int) -> Dict[str, Any]:
    """Ensure student_state has all required fields and defaults."""
    skills = student_state.get("skills") or {}
    last_practiced = student_state.get("last_practiced") or {}

    for topic in TOPICS:
        if topic not in skills:
            skills[topic] = dict(DEFAULT_SKILLS)
        else:
            for diff, base in DEFAULT_SKILLS.items():
                skills[topic].setdefault(diff, base)

        if topic not in last_practiced:
            last_practiced[topic] = dict(DEFAULT_LAST)
        else:
            for diff, base in DEFAULT_LAST.items():
                last_practiced[topic].setdefault(diff, base)

    student_state["skills"] = skills
    student_state["last_practiced"] = last_practiced
    student_state.setdefault("current_step", current_step)
    return student_state


def _apply_forgetting(student_state: Dict[str, Any], current_step: int) -> None:
    """Apply exponential forgetting to all skill entries in-place."""
    skills = student_state["skills"]
    last_practiced = student_state["last_practiced"]

    for topic in TOPICS:
        for diff in DIFFICULTIES:
            t_last = last_practiced.get(topic, {}).get(diff, 0)
            dt = max(current_step - t_last, 0)
            decay = math.exp(-dt / TAU) if dt > 0 else 1.0
            skills[topic][diff] *= decay


def _apply_previous_feedback(
    student_state: Dict[str, Any],
    current_step: int,
    previous_feedback: Dict[str, Any],
) -> None:
    """Update skills based on previous question feedback, in-place."""
    if not previous_feedback or not previous_feedback.get("was_previous_question", False):
        return

    prev_topic = previous_feedback.get("previous_topic")
    prev_diff_raw = previous_feedback.get("previous_difficulty")

    # Expect numeric difficulty 1-10 from the controller; try to coerce safely.
    try:
        prev_diff = int(prev_diff_raw)
    except (TypeError, ValueError):
        return

    if prev_topic not in TOPICS or prev_diff not in DIFFICULTIES:
        return

    skills = student_state["skills"]
    last_practiced = student_state["last_practiced"]

    old_skill = skills[prev_topic][prev_diff]
    r = 1.0 if previous_feedback.get("previous_correct", False) else 0.0
    new_skill = (1.0 - ALPHA) * old_skill + ALPHA * r
    skills[prev_topic][prev_diff] = new_skill

    # Assume previous question occurred at step current_step - 1
    last_practiced[prev_topic][prev_diff] = max(current_step - 1, 0)


def _choose_answer(
    skills: Dict[str, Any],
    topic: str,
    difficulty: int,
    question: str,
    options: Dict[str, str],
    rng: random.Random,
) -> (str, str):
    """Pick an answer and reasoning string.

    This deliberately behaves like a small, fallible model: it mostly guesses,
    with confidence style based on the skill level, and a couple of weak
    text-based heuristics so it isn't purely uniform random.
    """
    s = skills.get(topic, {}).get(difficulty, 0.05)

    # We always restrict answers to the canonical multiple-choice set A-D.
    valid_letters = ["A", "B", "C", "D"]

    # Very weak heuristic: prefer "all of the above" / "none of the above" style answers
    heuristic_choice = None
    for key in valid_letters:
        text = options.get(key, "")
        lower = text.lower()
        if "all of the above" in lower or "all of these" in lower:
            heuristic_choice = key
            break
        if "none of the above" in lower or "none of these" in lower:
            heuristic_choice = key
            break

    # Decision policy: mix heuristic with random, weighted by skill
    keys = [k for k in valid_letters if k in options] or valid_letters

    if heuristic_choice is not None and heuristic_choice in keys and rng.random() < min(max(s, 0.1), 0.7):
        ans = heuristic_choice
    else:
        ans = rng.choice(keys)

    # Build a short reasoning string reflecting (approximate) confidence
    if s < 0.2:
        reasoning = (
            f"This is a level {difficulty} question for {topic}, and my skill is still quite low "
            "here, so I'm mostly guessing based on a rough impression of the options."
        )
    elif s < 0.6:
        reasoning = (
            f"This is a level {difficulty} question for {topic}. I have some intuition, so I looked "
            "for small clues in the wording, but I'm still a bit unsure."
        )
    else:
        reasoning = (
            f"This is a level {difficulty} question for {topic}, and I feel relatively confident, "
            "so I picked the option that seems most consistent with the prompt, though I could "
            "still be wrong."
        )

    return ans, reasoning


def run_student_step(
    student_state: Dict[str, Any],
    previous_feedback: Dict[str, Any],
    current_task: Dict[str, Any],
    *,
    rng_seed: int | None = None,
) -> Dict[str, Any]:
    """Single step of the StudentAgent-SLM.

    Args:
        student_state: JSON-like state dict with skills/last_practiced/current_step.
        previous_feedback: Dict describing previous question outcome.
        current_task: Dict with keys 'topic', 'difficulty', 'question', 'options'.
        rng_seed: Optional seed for reproducible stochastic behavior.

    Returns:
        Dict with keys 'answer', 'reasoning', 'updated_state' (JSON-serializable).
    """
    # Determine current step from controller or default to 1
    t_now = int(student_state.get("current_step", 0))
    # Controller is expected to pass the *current* step; if not, bump it
    current_step = max(t_now, 1)

    rng = random.Random(rng_seed)

    # 1) Ensure full state structure
    state = _ensure_state(dict(student_state), current_step=current_step)

    # 2) Apply forgetting to all skills
    _apply_forgetting(state, current_step=current_step)

    # 3) Apply learning from previous feedback
    _apply_previous_feedback(state, current_step=current_step, previous_feedback=previous_feedback)

    # 4) Answer current question
    topic = current_task.get("topic", "logic")
    # Expect difficulty on a 1-10 numeric scale
    raw_diff = current_task.get("difficulty", 1)
    try:
        difficulty = int(raw_diff)
    except (TypeError, ValueError):
        difficulty = 1
    question = current_task.get("question", "")
    options = current_task.get("options", {}) or {}

    answer, reasoning = _choose_answer(
        state["skills"], topic, difficulty, question, options, rng
    )

    # 5) Advance step for next call
    state["current_step"] = current_step + 1

    return {
        "answer": answer,
        "reasoning": reasoning,
        "updated_state": state,
    }
