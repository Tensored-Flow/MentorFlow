"""
Task Generator for TeachRL
5 coding microtask families × 3 difficulty levels = 15 task types
Each task produces: obs_vec, correct_action, choices_vec, human_prompt, human_choices
"""

import random
from dataclasses import dataclass
from typing import List, Tuple, Callable

# Constants
NUM_FAMILIES = 5
NUM_DIFFICULTIES = 3
NUM_CHOICES = 4
OBS_VEC_SIZE = 32  # Fixed observation vector size

FAMILY_NAMES = ["var_trace", "if_cond", "loop_count", "list_index", "bool_logic"]
DIFFICULTY_NAMES = ["easy", "medium", "hard"]


@dataclass
class TaskSpec:
    """A single generated task instance."""
    family_id: int          # 0-4
    difficulty_id: int      # 0-2
    obs_vec: List[float]    # Fixed-length numerical encoding
    correct_action: int     # 0-3
    choices_vec: List[float]  # Flattened numeric choices
    human_prompt: str       # Pretty string for UI
    human_choices: List[str]  # Readable choice strings


# =============================================================================
# FAMILY 0: var_trace — Track variable values through assignments
# =============================================================================

def generate_var_trace(difficulty: int, rng: random.Random) -> TaskSpec:
    """
    Easy: 2 variables, 2 steps
    Medium: 3 variables, 3 steps
    Hard: 4 variables, 4 steps with expressions
    """
    if difficulty == 0:  # Easy
        a = rng.randint(1, 9)
        b = rng.randint(1, 9)
        # a = a + b
        result = a + b
        prompt = f"a = {a}\nb = {b}\na = a + b\nWhat is a?"
        var_encoding = [a, b, 0, 0, 1, 0, 0, 0]  # op: add
        
    elif difficulty == 1:  # Medium
        a = rng.randint(1, 9)
        b = rng.randint(1, 9)
        c = rng.randint(1, 9)
        # a = b + c, then b = a - c
        a_new = b + c
        b_new = a_new - c
        result = b_new
        prompt = f"a = {a}\nb = {b}\nc = {c}\na = b + c\nb = a - c\nWhat is b?"
        var_encoding = [a, b, c, 0, 1, 2, 0, 0]  # ops encoded
        
    else:  # Hard
        a = rng.randint(1, 5)
        b = rng.randint(1, 5)
        c = rng.randint(1, 5)
        d = rng.randint(1, 5)
        # a = b * c, d = a + d
        a_new = b * c
        d_new = a_new + d
        result = d_new
        prompt = f"a = {a}\nb = {b}\nc = {c}\nd = {d}\na = b * c\nd = a + d\nWhat is d?"
        var_encoding = [a, b, c, d, 2, 1, 0, 0]  # op: mul, add
    
    # Generate distractors
    correct = result
    distractors = _generate_numeric_distractors(correct, rng, count=3, min_val=0, max_val=50)
    choices, correct_idx = _shuffle_with_correct(correct, distractors, rng)
    
    obs_vec = _build_obs_vec(0, difficulty, var_encoding)
    choices_vec = [float(c) for c in choices]
    human_choices = [str(c) for c in choices]
    
    return TaskSpec(
        family_id=0,
        difficulty_id=difficulty,
        obs_vec=obs_vec,
        correct_action=correct_idx,
        choices_vec=choices_vec,
        human_prompt=prompt,
        human_choices=human_choices
    )


# =============================================================================
# FAMILY 1: if_cond — Evaluate conditional expressions
# =============================================================================

def generate_if_cond(difficulty: int, rng: random.Random) -> TaskSpec:
    """
    Easy: Simple comparison (x > y)
    Medium: Compound condition (x > y and z < w)
    Hard: Nested or chained conditions
    """
    if difficulty == 0:  # Easy
        x = rng.randint(1, 20)
        y = rng.randint(1, 20)
        op = rng.choice(['>', '<', '==', '>=', '<='])
        result = _eval_comparison(x, y, op)
        prompt = f"x = {x}\ny = {y}\nWhat is (x {op} y)?"
        cond_encoding = [x, y, 0, 0, _op_to_num(op), 0, 0, 0]
        
    elif difficulty == 1:  # Medium
        x, y = rng.randint(1, 15), rng.randint(1, 15)
        z, w = rng.randint(1, 15), rng.randint(1, 15)
        op1, op2 = rng.choice(['>', '<']), rng.choice(['>', '<'])
        logic = rng.choice(['and', 'or'])
        r1 = _eval_comparison(x, y, op1)
        r2 = _eval_comparison(z, w, op2)
        result = (r1 and r2) if logic == 'and' else (r1 or r2)
        prompt = f"x={x}, y={y}, z={z}, w={w}\nWhat is (x {op1} y) {logic} (z {op2} w)?"
        cond_encoding = [x, y, z, w, _op_to_num(op1), _op_to_num(op2), 1 if logic == 'and' else 2, 0]
        
    else:  # Hard
        a, b, c = rng.randint(1, 10), rng.randint(1, 10), rng.randint(1, 10)
        # (a > b) and (b > c) or (a == c)
        r1 = a > b
        r2 = b > c
        r3 = a == c
        result = (r1 and r2) or r3
        prompt = f"a={a}, b={b}, c={c}\nWhat is ((a > b) and (b > c)) or (a == c)?"
        cond_encoding = [a, b, c, 0, 1, 1, 3, 0]  # complex encoding
    
    correct = result
    choices = [True, False]
    correct_idx = 0 if correct else 1
    # Pad choices to 4
    choices_display = ["True", "False", "Error", "None"]
    choices_vec = [1.0, 0.0, -1.0, -2.0]
    
    obs_vec = _build_obs_vec(1, difficulty, cond_encoding)
    
    return TaskSpec(
        family_id=1,
        difficulty_id=difficulty,
        obs_vec=obs_vec,
        correct_action=correct_idx,
        choices_vec=choices_vec,
        human_prompt=prompt,
        human_choices=choices_display
    )


# =============================================================================
# FAMILY 2: loop_count — Count loop iterations or final value
# =============================================================================

def generate_loop_count(difficulty: int, rng: random.Random) -> TaskSpec:
    """
    Easy: Simple for loop count
    Medium: While loop with condition
    Hard: Nested loop or accumulator
    """
    if difficulty == 0:  # Easy
        start, end = 0, rng.randint(3, 8)
        result = end - start  # iterations
        prompt = f"count = 0\nfor i in range({start}, {end}):\n    count += 1\nWhat is count?"
        loop_encoding = [start, end, 0, 0, 1, 0, 0, 0]
        
    elif difficulty == 1:  # Medium
        x = rng.randint(10, 20)
        step = rng.randint(2, 4)
        count = 0
        val = x
        while val > 0:
            val -= step
            count += 1
        result = count
        prompt = f"x = {x}\ncount = 0\nwhile x > 0:\n    x -= {step}\n    count += 1\nWhat is count?"
        loop_encoding = [x, step, 0, 0, 2, 0, 0, 0]
        
    else:  # Hard
        n = rng.randint(2, 4)
        m = rng.randint(2, 4)
        result = n * m  # nested loop iterations
        prompt = f"count = 0\nfor i in range({n}):\n    for j in range({m}):\n        count += 1\nWhat is count?"
        loop_encoding = [n, m, 0, 0, 3, 0, 0, 0]
    
    correct = result
    distractors = _generate_numeric_distractors(correct, rng, count=3, min_val=0, max_val=30)
    choices, correct_idx = _shuffle_with_correct(correct, distractors, rng)
    
    obs_vec = _build_obs_vec(2, difficulty, loop_encoding)
    choices_vec = [float(c) for c in choices]
    human_choices = [str(c) for c in choices]
    
    return TaskSpec(
        family_id=2,
        difficulty_id=difficulty,
        obs_vec=obs_vec,
        correct_action=correct_idx,
        choices_vec=choices_vec,
        human_prompt=prompt,
        human_choices=human_choices
    )


# =============================================================================
# FAMILY 3: list_index — List indexing and access
# =============================================================================

def generate_list_index(difficulty: int, rng: random.Random) -> TaskSpec:
    """
    Easy: Direct positive index
    Medium: Negative index
    Hard: Nested list or slice length
    """
    if difficulty == 0:  # Easy
        lst = [rng.randint(1, 9) for _ in range(5)]
        idx = rng.randint(0, 4)
        result = lst[idx]
        prompt = f"lst = {lst}\nWhat is lst[{idx}]?"
        list_encoding = lst + [idx, 0, 0]
        
    elif difficulty == 1:  # Medium
        lst = [rng.randint(1, 9) for _ in range(5)]
        neg_idx = rng.randint(-5, -1)
        result = lst[neg_idx]
        prompt = f"lst = {lst}\nWhat is lst[{neg_idx}]?"
        list_encoding = lst + [neg_idx + 10, 1, 0]  # offset negative
        
    else:  # Hard
        lst = [[rng.randint(1, 5) for _ in range(3)] for _ in range(3)]
        i, j = rng.randint(0, 2), rng.randint(0, 2)
        result = lst[i][j]
        flat = [item for sublist in lst for item in sublist][:8]
        prompt = f"lst = {lst}\nWhat is lst[{i}][{j}]?"
        list_encoding = flat if len(flat) >= 8 else flat + [0] * (8 - len(flat))
    
    correct = result
    distractors = _generate_numeric_distractors(correct, rng, count=3, min_val=1, max_val=9)
    choices, correct_idx = _shuffle_with_correct(correct, distractors, rng)
    
    obs_vec = _build_obs_vec(3, difficulty, list_encoding[:8])
    choices_vec = [float(c) for c in choices]
    human_choices = [str(c) for c in choices]
    
    return TaskSpec(
        family_id=3,
        difficulty_id=difficulty,
        obs_vec=obs_vec,
        correct_action=correct_idx,
        choices_vec=choices_vec,
        human_prompt=prompt,
        human_choices=human_choices
    )


# =============================================================================
# FAMILY 4: bool_logic — Boolean expression evaluation
# =============================================================================

def generate_bool_logic(difficulty: int, rng: random.Random) -> TaskSpec:
    """
    Easy: 2 operands with single operator
    Medium: 3 operands with two operators
    Hard: 4 operands with mixed operators
    """
    if difficulty == 0:  # Easy
        a, b = rng.choice([True, False]), rng.choice([True, False])
        op = rng.choice(['and', 'or'])
        result = (a and b) if op == 'and' else (a or b)
        prompt = f"a = {a}\nb = {b}\nWhat is (a {op} b)?"
        bool_encoding = [int(a), int(b), 0, 0, 1 if op == 'and' else 2, 0, 0, 0]
        
    elif difficulty == 1:  # Medium
        a, b, c = [rng.choice([True, False]) for _ in range(3)]
        op1, op2 = rng.choice(['and', 'or']), rng.choice(['and', 'or'])
        # (a op1 b) op2 c
        r1 = (a and b) if op1 == 'and' else (a or b)
        result = (r1 and c) if op2 == 'and' else (r1 or c)
        prompt = f"a={a}, b={b}, c={c}\nWhat is ((a {op1} b) {op2} c)?"
        bool_encoding = [int(a), int(b), int(c), 0, 
                         1 if op1 == 'and' else 2, 
                         1 if op2 == 'and' else 2, 0, 0]
        
    else:  # Hard
        a, b, c, d = [rng.choice([True, False]) for _ in range(4)]
        # (a and b) or (c and d)
        result = (a and b) or (c and d)
        prompt = f"a={a}, b={b}, c={c}, d={d}\nWhat is ((a and b) or (c and d))?"
        bool_encoding = [int(a), int(b), int(c), int(d), 1, 2, 1, 0]
    
    correct = result
    choices_display = ["True", "False", "Error", "None"]
    choices_vec = [1.0, 0.0, -1.0, -2.0]
    correct_idx = 0 if correct else 1
    
    obs_vec = _build_obs_vec(4, difficulty, bool_encoding)
    
    return TaskSpec(
        family_id=4,
        difficulty_id=difficulty,
        obs_vec=obs_vec,
        correct_action=correct_idx,
        choices_vec=choices_vec,
        human_prompt=prompt,
        human_choices=choices_display
    )


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _build_obs_vec(family_id: int, difficulty_id: int, task_encoding: List[float]) -> List[float]:
    """Build fixed-length observation vector."""
    # One-hot family (5) + one-hot difficulty (3) + task encoding (padded to 24)
    obs = [0.0] * OBS_VEC_SIZE
    
    # One-hot family encoding (indices 0-4)
    obs[family_id] = 1.0
    
    # One-hot difficulty encoding (indices 5-7)
    obs[5 + difficulty_id] = 1.0
    
    # Task-specific encoding (indices 8-31)
    for i, val in enumerate(task_encoding[:24]):
        obs[8 + i] = float(val)
    
    return obs


def _op_to_num(op: str) -> int:
    """Convert comparison operator to numeric."""
    ops = {'<': 1, '>': 2, '==': 3, '<=': 4, '>=': 5, '!=': 6}
    return ops.get(op, 0)


def _eval_comparison(x: int, y: int, op: str) -> bool:
    """Evaluate comparison."""
    if op == '<': return x < y
    if op == '>': return x > y
    if op == '==': return x == y
    if op == '<=': return x <= y
    if op == '>=': return x >= y
    if op == '!=': return x != y
    return False


def _generate_numeric_distractors(correct: int, rng: random.Random, 
                                   count: int = 3, min_val: int = 0, 
                                   max_val: int = 50) -> List[int]:
    """Generate plausible wrong answers."""
    distractors = set()
    attempts = 0
    while len(distractors) < count and attempts < 100:
        # Generate near the correct answer
        offset = rng.randint(-5, 5)
        if offset == 0:
            offset = rng.choice([-1, 1])
        candidate = correct + offset
        if candidate != correct and min_val <= candidate <= max_val:
            distractors.add(candidate)
        attempts += 1
    
    # Fill remaining with random if needed
    while len(distractors) < count:
        candidate = rng.randint(min_val, max_val)
        if candidate != correct:
            distractors.add(candidate)
    
    return list(distractors)[:count]


def _shuffle_with_correct(correct, distractors: List, rng: random.Random) -> Tuple[List, int]:
    """Shuffle correct answer with distractors, return choices and correct index."""
    choices = [correct] + distractors
    rng.shuffle(choices)
    correct_idx = choices.index(correct)
    return choices, correct_idx


# =============================================================================
# MAIN API
# =============================================================================

GENERATORS: List[Callable] = [
    generate_var_trace,
    generate_if_cond,
    generate_loop_count,
    generate_list_index,
    generate_bool_logic,
]


def generate_task(family_id: int, difficulty_id: int, seed: int = None) -> TaskSpec:
    """
    Generate a single task.
    
    Args:
        family_id: 0-4 (var_trace, if_cond, loop_count, list_index, bool_logic)
        difficulty_id: 0-2 (easy, medium, hard)
        seed: Optional seed for deterministic generation
    
    Returns:
        TaskSpec with all task information
    """
    if not (0 <= family_id < NUM_FAMILIES):
        raise ValueError(f"family_id must be 0-{NUM_FAMILIES-1}")
    if not (0 <= difficulty_id < NUM_DIFFICULTIES):
        raise ValueError(f"difficulty_id must be 0-{NUM_DIFFICULTIES-1}")
    
    rng = random.Random(seed)
    return GENERATORS[family_id](difficulty_id, rng)


def generate_task_by_arm(arm_id: int, seed: int = None) -> TaskSpec:
    """
    Generate task by arm index (for bandit teacher).
    arm_id = family_id * 3 + difficulty_id
    """
    if not (0 <= arm_id < NUM_FAMILIES * NUM_DIFFICULTIES):
        raise ValueError(f"arm_id must be 0-{NUM_FAMILIES * NUM_DIFFICULTIES - 1}")
    
    family_id = arm_id // NUM_DIFFICULTIES
    difficulty_id = arm_id % NUM_DIFFICULTIES
    return generate_task(family_id, difficulty_id, seed)


def generate_eval_dataset(tasks_per_type: int = 10, seed: int = 42) -> List[TaskSpec]:
    """
    Generate a fixed evaluation dataset.
    
    Args:
        tasks_per_type: Number of tasks per (family, difficulty) pair
        seed: Seed for reproducibility
    
    Returns:
        List of TaskSpecs (15 * tasks_per_type total)
    """
    rng = random.Random(seed)
    dataset = []
    
    for family_id in range(NUM_FAMILIES):
        for difficulty_id in range(NUM_DIFFICULTIES):
            for i in range(tasks_per_type):
                task_seed = rng.randint(0, 2**31 - 1)
                task = generate_task(family_id, difficulty_id, task_seed)
                dataset.append(task)
    
    return dataset


def arm_to_name(arm_id: int) -> str:
    """Convert arm_id to human-readable name."""
    family_id = arm_id // NUM_DIFFICULTIES
    difficulty_id = arm_id % NUM_DIFFICULTIES
    return f"{FAMILY_NAMES[family_id]}_{DIFFICULTY_NAMES[difficulty_id]}"


# =============================================================================
# STANDALONE TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("TeachRL Task Generator Test")
    print("=" * 60)
    
    # Test each family at each difficulty
    for fam in range(NUM_FAMILIES):
        for diff in range(NUM_DIFFICULTIES):
            task = generate_task(fam, diff, seed=123)
            print(f"\n[{FAMILY_NAMES[fam]}][{DIFFICULTY_NAMES[diff]}]")
            print(f"Prompt:\n{task.human_prompt}")
            print(f"Choices: {task.human_choices}")
            print(f"Correct: {task.human_choices[task.correct_action]}")
            print(f"obs_vec length: {len(task.obs_vec)}")
    
    # Test eval dataset
    print("\n" + "=" * 60)
    eval_set = generate_eval_dataset(tasks_per_type=2, seed=42)
    print(f"Eval dataset size: {len(eval_set)}")
    print("Reproducibility check:", generate_task(0, 0, 999).correct_action == 
                                    generate_task(0, 0, 999).correct_action)
