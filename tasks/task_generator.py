"""
Task Generator for MentorFlow

18 task families × 5 difficulty levels = 90 task types.
Fully procedural generation (no static datasets) with human-readable prompts
and fixed-size observation vectors for the student agent.

Difficulty levels (1–5):
1. Beginner (recall / direct evaluation)
2. Intermediate (2-step reasoning)
3. Advanced (multi-step logic)
4. Expert (edge cases / tricky flows)
5. Olympiad++ (multi-step inference + non-standard traps)
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, List, Tuple

NUM_FAMILIES = 18
NUM_DIFFICULTIES = 5  # 1–5 inclusive
NUM_CHOICES = 4
OBS_VEC_SIZE = 32

FAMILY_NAMES = [
    "var_trace",
    "if_cond",
    "loop_count",
    "list_index",
    "bool_logic",
    "python_syntax",
    "java_syntax",
    "js_syntax",
    "cpp_syntax",
    "rust_syntax",
    "sql_syntax",
    "html_css",
    "bash_syntax",
    "regex",
    "math_algebra",
    "physics_kinematics",
    "chemistry",
    "probability",
]

DIFFICULTY_NAMES = [
    "beginner",
    "intermediate",
    "advanced",
    "expert",
    "olympiad_plus",
]


@dataclass(slots=True)
class TaskSpec:
    """A single generated task instance."""

    family_id: int
    difficulty_id: int  # 1–5 for UX clarity
    obs_vec: List[float]
    correct_action: int
    choices_vec: List[float]
    human_prompt: str
    human_choices: List[str]


# =============================================================================
# UTILS
# =============================================================================


def _normalize_difficulty(level: int) -> int:
    if level < 1:
        level += 1  # allow legacy 0-based callers
    return max(1, min(NUM_DIFFICULTIES, level))


def _shuffle_with_correct(correct, distractors: List, rng: random.Random) -> Tuple[List, int]:
    choices = [correct] + distractors
    rng.shuffle(choices)
    return choices, choices.index(correct)


def _generate_numeric_distractors(correct: float, rng: random.Random, spread: int = 4) -> List[float]:
    candidates = set()
    deltas = list(range(-spread, spread + 1))
    deltas = [d for d in deltas if d != 0]
    while len(candidates) < 3:
        delta = rng.choice(deltas)
        candidate = correct + delta
        if candidate != correct:
            candidates.add(candidate)
    return list(candidates)[:3]


def _to_numeric_choice(choice) -> float:
    if isinstance(choice, bool):
        return 1.0 if choice else 0.0
    try:
        return float(choice)
    except (TypeError, ValueError):
        return float(len(str(choice)))


def _encode_text_features(prompt: str, choices: List[str]) -> List[float]:
    """Lightweight text stats to keep obs vec informative without being huge."""
    tokens = prompt.split()
    token_count = len(tokens)
    digit_count = sum(ch.isdigit() for ch in prompt)
    upper_ratio = sum(ch.isupper() for ch in prompt) / max(1, len(prompt))
    lower_ratio = sum(ch.islower() for ch in prompt) / max(1, len(prompt))
    punctuation = sum(prompt.count(sym) for sym in ["?", ":", ",", "(", ")", "[", "]"])
    choice_lengths = [len(c) for c in choices] or [0]

    features = [
        len(prompt) / 256.0,
        token_count / 64.0,
        digit_count / max(1.0, len(prompt)),
        upper_ratio,
        lower_ratio,
        punctuation / 32.0,
        len(choices) / 8.0,
        sum(choice_lengths) / max(1.0, 128.0),
        max(choice_lengths) / max(1.0, 96.0),
        sum(1 for c in choices if c.lower() in {"true", "false"}) / 4.0,
        sum(1 for c in choices if c.isdigit()) / 8.0,
    ]

    # Pad / trim to fit into OBS_VEC_SIZE - 4
    target = OBS_VEC_SIZE - 4
    if len(features) < target:
        features.extend([0.0] * (target - len(features)))
    return features[:target]


def _build_obs_vec(family_id: int, difficulty: int, prompt: str, choices: List[str]) -> List[float]:
    """Compact observation vector: meta features + text stats."""
    obs = [0.0] * OBS_VEC_SIZE
    obs[0] = family_id / max(1, NUM_FAMILIES - 1)
    obs[1] = difficulty / NUM_DIFFICULTIES
    obs[2] = (family_id + 1) / NUM_FAMILIES  # second view of family position
    obs[3] = len(choices) / NUM_CHOICES

    encoded = _encode_text_features(prompt, choices)
    for i, val in enumerate(encoded):
        obs[4 + i] = float(val)
    return obs


def _task_from_values(
    family_id: int,
    difficulty: int,
    prompt: str,
    correct,
    distractors: List,
    rng: random.Random,
) -> TaskSpec:
    choices, idx = _shuffle_with_correct(correct, distractors, rng)
    human_choices = [str(c) for c in choices]
    obs_vec = _build_obs_vec(family_id, difficulty, prompt, human_choices)
    choices_vec = [_to_numeric_choice(c) for c in choices]
    return TaskSpec(
        family_id=family_id,
        difficulty_id=difficulty,
        obs_vec=obs_vec,
        correct_action=idx,
        choices_vec=choices_vec,
        human_prompt=prompt,
        human_choices=human_choices,
    )


# =============================================================================
# FAMILY 0: var_trace — Track variable values through assignments
# =============================================================================


def generate_var_trace(level: int, rng: random.Random) -> TaskSpec:
    level = _normalize_difficulty(level)
    variant = rng.randint(0, 2)  # pick among a few lightweight templates

    if level == 1:
        if variant == 0:
            a = rng.randint(2, 9)
            b = rng.randint(2, 9)
            prompt = f"a = {a}\nb = {b}\na = a + b\nWhat is a?"
            result = a + b
        elif variant == 1:
            a = rng.randint(3, 7)
            b = rng.randint(1, 4)
            prompt = f"a = {a}\na += {b}\na -= 1\nWhat is a?"
            result = a + b - 1
        else:
            a = rng.randint(2, 6)
            b = rng.randint(1, 3)
            prompt = f"a = {a}\na = a * 2 - {b}\nWhat is a?"
            result = a * 2 - b
    elif level == 2:
        if variant == 0:
            x = rng.randint(3, 8)
            y = x + rng.randint(2, 5)
            z = y - rng.randint(1, 3)
            w = z * 2 - x
            prompt = f"x = {x}\ny = x + {y - x}\nz = y - {y - z}\nw = z * 2 - x\nWhat is w?"
            result = w
        elif variant == 1:
            x = rng.randint(2, 6)
            y = rng.randint(2, 6)
            z = rng.randint(1, 4)
            prompt = (
                f"x = {x}\ny = {y}\n"
                "for i in range(2):\n"
                "    x = x + y\n"
                f"    y = y - {z}\n"
                "What is x?"
            )
            result = (x + y) + (x + y - z)
        else:
            base = rng.randint(2, 6)
            inc = rng.randint(1, 3)
            dec = rng.randint(1, 2)
            prompt = f"val = {base}\nval += {inc}\nval *= 2\nval -= {dec}\nWhat is val?"
            result = ((base + inc) * 2) - dec
    elif level == 3:
        if variant == 0:
            base = rng.randint(2, 6)
            inc = rng.randint(2, 4)
            dec = rng.randint(1, 3)
            a = base + inc
            b = (a - dec) * 2
            c = b + (a % dec)
            prompt = f"a = {base} + {inc}\nb = (a - {dec}) * 2\nc = b + (a % {dec})\nWhat is c?"
            result = c
        elif variant == 1:
            vals = [rng.randint(1, 5) for _ in range(3)]
            prompt = (
                f"a, b, c = {vals}\n"
                "a = a + b\n"
                "b = b + c\n"
                "c = a + b\n"
                "What is c?"
            )
            a, b, c = vals
            a = a + b
            b = b + c
            c = a + b
            result = c
        else:
            nums = [rng.randint(1, 4) for _ in range(4)]
            prompt = (
                f"nums = {nums}\n"
                "total = 0\n"
                "for i, v in enumerate(nums):\n"
                "    total += v * (i+1)\n"
                "What is total?"
            )
            result = sum(v * (i + 1) for i, v in enumerate(nums))
    elif level == 4:
        seq = [rng.randint(1, 9) for _ in range(4)]
        acc = 0
        for i, val in enumerate(seq):
            if i % 2 == 0:
                acc += val * (i + 1)
            else:
                acc -= val // (i + 1)
        acc += seq[0]
        prompt = (
            f"seq = {seq}\nacc = 0\n"
            "for i, v in enumerate(seq):\n"
            "    if i % 2 == 0:\n"
            "        acc += v * (i+1)\n"
            "    else:\n"
            "        acc -= v // (i+1)\n"
            "acc += seq[0]\nWhat is acc?"
        )
        result = acc
    else:
        grid = [[rng.randint(1, 5) for _ in range(3)] for _ in range(2)]
        carry = rng.randint(1, 3)
        for i, row in enumerate(grid):
            for j, val in enumerate(row):
                if (i + j) % 2 == 0:
                    carry += val + j
                else:
                    carry -= max(1, val - i)
            grid[i][0] += carry % 3
        result = carry + grid[1][0] - grid[0][-1]
        prompt = (
            f"grid = {grid}\ncarry = {carry % 5}\n"
            "for i, row in enumerate(grid):\n"
            "    for j, v in enumerate(row):\n"
            "        if (i + j) % 2 == 0:\n"
            "            carry += v + j\n"
            "        else:\n"
            "            carry -= max(1, v - i)\n"
            "    grid[i][0] += carry % 3\n"
            "result = carry + grid[1][0] - grid[0][-1]\nWhat is result?"
        )
    distractors = _generate_numeric_distractors(result, rng)
    return _task_from_values(0, level, prompt, result, distractors, rng)


# =============================================================================
# FAMILY 1: if_cond — Evaluate conditional expressions
# =============================================================================


def generate_if_cond(level: int, rng: random.Random) -> TaskSpec:
    level = _normalize_difficulty(level)
    variant = rng.randint(0, 2)
    choices = ["True", "False", "Depends", "Error"]

    if level == 1:
        if variant == 0:
            x = rng.randint(1, 9)
            y = x + rng.randint(1, 4)
            prompt = f"x = {x}\ny = {y}\nIs x < y?"
            result = x < y
        else:
            a = rng.randint(1, 5)
            b = rng.randint(1, 5)
            prompt = f"a = {a}\nb = {b}\nIs (a * 2) >= (b + 1)?"
            result = (a * 2) >= (b + 1)
    elif level == 2:
        if variant == 0:
            a = rng.randint(2, 6)
            b = rng.randint(1, 4)
            c = rng.randint(3, 9)
            prompt = f"a = {a}\nb = {b}\nc = {c}\nIs (a + b > c) and (c % 2 == 1)?"
            result = (a + b > c) and (c % 2 == 1)
        else:
            x = rng.randint(2, 6)
            y = rng.randint(2, 6)
            z = rng.randint(1, 4)
            prompt = f"x={x}, y={y}, z={z}\nIs (x - z < y) or (y % 2 == 0)?"
            result = (x - z < y) or (y % 2 == 0)
    elif level == 3:
        x = rng.randint(2, 6)
        y = rng.randint(2, 6)
        z = rng.randint(5, 12)
        prompt = f"x = {x}\ny = {y}\nz = {z}\nEvaluate: (x == y) or (z < x + y) and not(x > z)"
        result = (x == y) or (z < x + y) and not (x > z)
    elif level == 4:
        nums = [rng.randint(1, 6) for _ in range(4)]
        cond = (all(n % 2 == 0 for n in nums[:2]) or any(n > 4 for n in nums[2:])) and not (nums[0] == nums[-1])
        prompt = (
            f"nums = {nums}\n"
            "Evaluate: (all even for first two OR any > 4 in last two) and nums[0] != nums[-1]"
        )
        result = cond
    else:
        a = rng.randint(1, 3)
        b = rng.randint(3, 7)
        c = rng.randint(2, 5)
        expr = ((a < b == c) or ((b - a) and not (c % 2)) or (a + b > c * 2)) and not (b == c == 0)
        prompt = (
            f"a = {a}, b = {b}, c = {c}\n"
            "Evaluate with Python precedence:\n"
            "(a < b == c) or ((b - a) and not (c % 2)) or (a + b > c * 2) and not (b == c == 0)"
        )
        result = expr

    correct_idx = 0 if result else 1
    obs_vec = _build_obs_vec(1, level, prompt, choices)
    choices_vec = [1.0, 0.0, -1.0, -2.0]
    return TaskSpec(
        family_id=1,
        difficulty_id=level,
        obs_vec=obs_vec,
        correct_action=correct_idx,
        choices_vec=choices_vec,
        human_prompt=prompt,
        human_choices=choices,
    )


# =============================================================================
# FAMILY 2: loop_count — Count loop iterations or final value
# =============================================================================


def generate_loop_count(level: int, rng: random.Random) -> TaskSpec:
    level = _normalize_difficulty(level)
    variant = rng.randint(0, 2)

    if level == 1:
        if variant == 0:
            n = rng.randint(2, 5)
            prompt = f"count = 0\nfor i in range({n}):\n    count += 1\nWhat is count?"
            result = n
        else:
            n = rng.randint(2, 5)
            start = rng.randint(0, 2)
            prompt = f"count = {start}\nfor i in range({n}):\n    count += i\nWhat is count?"
            result = start + sum(range(n))
    elif level == 2:
        start = rng.randint(8, 14)
        step = rng.randint(2, 4)
        count = 0
        x = start
        while x > 0:
            x -= step
            count += 1
        prompt = (
            f"x = {start}\nstep = {step}\ncount = 0\nwhile x > 0:\n    x -= step\n    count += 1\nWhat is count?"
        )
        result = count
    elif level == 3:
        outer = rng.randint(2, 4)
        inner = rng.randint(2, 4)
        count = 0
        for i in range(outer):
            for j in range(inner):
                if (i + j) % 2 == 0:
                    count += 1
        prompt = (
            f"count = 0\nfor i in range({outer}):\n"
            f"    for j in range({inner}):\n"
            "        if (i + j) % 2 == 0:\n            count += 1\nWhat is count?"
        )
        result = count
    elif level == 4:
        limit = rng.randint(3, 6)
        skip = rng.randint(1, limit - 1)
        total = 0
        for i in range(1, limit + 2):
            if i == skip:
                continue
            for j in range(i):
                if (i * j) % 3 == 0:
                    total += 1
        prompt = (
            f"total = 0\nfor i in range(1, {limit + 2}):\n"
            f"    if i == {skip}: continue\n"
            "    for j in range(i):\n"
            "        if (i * j) % 3 == 0:\n"
            "            total += 1\nWhat is total?"
        )
        result = total
    else:
        data = [rng.randint(2, 6) for _ in range(4)]
        trips = 0
        for idx, val in enumerate(data):
            for j in range(val):
                if j % (idx + 2) == 0:
                    trips += 1
                elif (j + val) % 4 == 0:
                    trips -= 1
        prompt = (
            f"data = {data}\ntrips = 0\n"
            "for idx, val in enumerate(data):\n"
            "    for j in range(val):\n"
            "        if j % (idx+2) == 0:\n"
            "            trips += 1\n"
            "        elif (j + val) % 4 == 0:\n"
            "            trips -= 1\n"
            "What is trips?"
        )
        result = trips

    distractors = _generate_numeric_distractors(result, rng)
    return _task_from_values(2, level, prompt, result, distractors, rng)


# =============================================================================
# FAMILY 3: list_index — List indexing and access
# =============================================================================


def generate_list_index(level: int, rng: random.Random) -> TaskSpec:
    level = _normalize_difficulty(level)

    if level == 1:
        lst = [rng.randint(1, 9) for _ in range(5)]
        idx = rng.randint(0, 4)
        prompt = f"lst = {lst}\nWhat is lst[{idx}]?"
        result = lst[idx]
    elif level == 2:
        lst = [rng.randint(10, 40) for _ in range(6)]
        prompt = f"lst = {lst}\nWhat is lst[-2]?"
        result = lst[-2]
    elif level == 3:
        base = [[rng.randint(1, 6) for _ in range(3)] for _ in range(3)]
        i = rng.randint(0, 2)
        j = rng.randint(0, 2)
        prompt = f"lst = {base}\nWhat is lst[{i}][{j}]?"
        result = base[i][j]
    elif level == 4:
        lst = [rng.randint(-5, 5) for _ in range(7)]
        slice_part = lst[1:-1:2]
        prompt = f"lst = {lst}\npart = lst[1:-1:2]\npart[0] = part[0] + lst[-1]\nWhat is part[-1]?"
        slice_part[0] = slice_part[0] + lst[-1]
        result = slice_part[-1]
    else:
        lst = [rng.randint(-8, 12) for _ in range(8)]
        alias = lst[2:7]
        alias[1] = alias[1] + lst[-1]
        nested = alias[::-2]
        prompt = (
            f"lst = {lst}\n"
            "alias = lst[2:7]\n"
            "alias[1] = alias[1] + lst[-1]\n"
            "nested = alias[::-2]\n"
            "What is nested[1] + lst[2]?"
        )
        result = nested[1] + lst[2]

    distractors = _generate_numeric_distractors(result, rng)
    return _task_from_values(3, level, prompt, result, distractors, rng)


# =============================================================================
# FAMILY 4: bool_logic — Boolean expression evaluation
# =============================================================================


def generate_bool_logic(level: int, rng: random.Random) -> TaskSpec:
    level = _normalize_difficulty(level)
    choices = ["True", "False", "Needs more info", "Error"]

    if level == 1:
        prompt = "a = True\nb = False\nEvaluate: a and b"
        result = False
    elif level == 2:
        prompt = "x = True, y = False, z = True\nEvaluate: (x or y) and not z"
        result = False
    elif level == 3:
        prompt = "a=True, b=False, c=True\nEvaluate: (a and (not b)) xor (b or c)"
        left = True and (not False)
        right = False or True
        result = (left and not right) or (right and not left)
    elif level == 4:
        prompt = "p=1, q=0, r=2\nEvaluate: (p and q) or (r > 1) and not(q ^ r)"
        result = (1 and 0) or (2 > 1) and not (0 ^ 2)
    else:
        prompt = (
            "flags = [True, False, True]\n"
            "Evaluate: (flags[0] ^ flags[1]) and (not flags[1] or flags[2]) and not(flags.count(True) == 3)"
        )
        flags = [True, False, True]
        result = (flags[0] ^ flags[1]) and (not flags[1] or flags[2]) and not (flags.count(True) == 3)

    correct_idx = 0 if result else 1
    obs_vec = _build_obs_vec(4, level, prompt, choices)
    choices_vec = [1.0, 0.0, -1.0, -2.0]
    return TaskSpec(
        family_id=4,
        difficulty_id=level,
        obs_vec=obs_vec,
        correct_action=correct_idx,
        choices_vec=choices_vec,
        human_prompt=prompt,
        human_choices=choices,
    )


# =============================================================================
# FAMILY 5-9: language syntax variants
# =============================================================================


def _build_language_task(family_id: int, language: str, level: int, rng: random.Random) -> TaskSpec:
    level = _normalize_difficulty(level)

    if language == "python":
        if level == 1:
            lst = [rng.randint(1, 9) for _ in range(4)]
            prompt = f"lst = {lst}\nWhat is len(lst)?"
            result = len(lst)
        elif level == 2:
            prompt = "x = []\nEvaluate: bool(x)"
            result = False
        elif level == 3:
            nums = [rng.randint(1, 5) for _ in range(4)]
            comp = [n for n in nums if n % 2 == 0]
            prompt = f"nums = {nums}\ncomp = [n for n in nums if n % 2 == 0]\nWhat is comp[-1]?"
            result = comp[-1] if comp else 0
        elif level == 4:
            base = [1, 2, 3]
            alias = base
            # Iterate over a snapshot to avoid unbounded list growth while still demonstrating alias side effects
            comp = []
            for x in list(base):  # snapshot prevents infinite growth
                alias.append(0)  # mutates both alias and base
                comp.append(x * alias[0])
            prompt = (
                "base = [1, 2, 3]\n"
                "alias = base\n"
                "comp = []\n"
                "for x in list(base):\n"
                "    alias.append(0)\n"
                "    comp.append(x * alias[0])\n"
                "What is comp[-1]?"
            )
            result = comp[-1]
        else:
            nums = [2, 3, 4]
            gen = (n for n in nums if (nums.__setitem__(0, nums[0] + 1) or True))
            first = next(gen)
            second = next(gen)
            prompt = (
                "nums = [2, 3, 4]\n"
                "gen = (n for n in nums if (nums.__setitem__(0, nums[0] + 1) or True))\n"
                "first = next(gen); second = next(gen)\n"
                "What is first + second + nums[0]?"
            )
            result = first + second + nums[0]
    elif language == "java":
        if level == 1:
            prompt = "int x = 2 + 3; What is x?"
            result = 5
        elif level == 2:
            prompt = "boolean flag = (5 > 3) && (2 < 1); What is flag?"
            result = False
        elif level == 3:
            prompt = "String s = \"hi\" + 2 + 3; What is s?"
            result = "hi23"
        elif level == 4:
            prompt = (
                "int a = 3; int b = ++a * 2; int c = b++ + a; What is c?"
            )
            a = 3
            b = (a + 1) * 2
            a += 1
            c = b + a
            b += 1
            result = c
        else:
            prompt = (
                "int x = 2; int y = x++ + ++x; int z = y++ + x--; return z + x;"
            )
            x = 2
            y = x + 1 + (x + 2)
            x += 2
            z = y + x
            y += 1
            x -= 1
            result = z + x
    elif language == "javascript":
        if level == 1:
            prompt = "const x = 2 + 3; What is x?"
            result = 5
        elif level == 2:
            prompt = "What is Boolean([])?"
            result = True
        elif level == 3:
            prompt = "const x = '5' == 5; What is x?"
            result = True
        elif level == 4:
            prompt = "What does [] + {} evaluate to in JS?"
            result = "[object Object]"
        else:
            prompt = "Evaluate in JS: ([] + {}) == ({} + []) && (NaN == NaN)"
            left = (str([]) + str({})) == (str({}) + str([]))
            result = left and (float("nan") == float("nan"))
    elif language == "cpp":
        if level == 1:
            prompt = "int a = 2 + 4; What is a?"
            result = 6
        elif level == 2:
            prompt = "int x = 5; int y = x / 2; What is y?"
            result = 2
        elif level == 3:
            prompt = "int a = 3; int b = a++; What is b?"
            result = 3
        elif level == 4:
            prompt = (
                "int a=2; int b=a++; int &r=a; r+=b; int *p=&b; *p+=a; What is a+b?"
            )
            a = 2
            b = a
            a += 1
            a += b
            b += a
            result = a + b
        else:
            prompt = (
                "int a=3; int b=4; int& r=a; int* p=&b; r+=*p; *p+=r; int c=*p - a; What is c?"
            )
            a = 3
            b = 4
            r = a + b
            a = r
            b = b + a
            c = b - a
            result = c
    else:  # rust
        if level == 1:
            prompt = "let x = 2 + 5; What is x?"
            result = 7
        elif level == 2:
            prompt = "let s = String::from(\"hi\"); let len = s.len(); What is len?"
            result = 2
        elif level == 3:
            prompt = (
                "let mut v = vec![1,2,3]; let first = v[0]; v.push(4); What is v.len()?"
            )
            result = 4
        elif level == 4:
            prompt = (
                "let mut x = 3; { let y = &mut x; *y += 2; } x += 1; What is x?"
            )
            result = 6
        else:
            prompt = (
                "let mut v = vec![1,2]; let r = &v; let s = v.clone(); drop(r); v.push(s[0]); What is v.len()?"
            )
            result = 3

    if isinstance(result, (int, float, bool)):
        distractors = _generate_numeric_distractors(result, rng)
    else:
        base = str(result)
        distractors = [
            base.swapcase(),
            f"{base}!",
            "undefined" if base.lower() != "undefined" else "null",
        ]
    distractors = [d for d in distractors if d != result][:3]
    return _task_from_values(family_id, level, prompt, result, distractors, rng)


def generate_python_syntax(level: int, rng: random.Random) -> TaskSpec:
    return _build_language_task(5, "python", level, rng)


def generate_java_syntax(level: int, rng: random.Random) -> TaskSpec:
    return _build_language_task(6, "java", level, rng)


def generate_js_syntax(level: int, rng: random.Random) -> TaskSpec:
    return _build_language_task(7, "javascript", level, rng)


def generate_cpp_syntax(level: int, rng: random.Random) -> TaskSpec:
    return _build_language_task(8, "cpp", level, rng)


def generate_rust_syntax(level: int, rng: random.Random) -> TaskSpec:
    return _build_language_task(9, "rust", level, rng)


# =============================================================================
# FAMILY 10: sql_syntax — joins and filters
# =============================================================================


def generate_sql_syntax(level: int, rng: random.Random) -> TaskSpec:
    level = _normalize_difficulty(level)

    if level == 1:
        prompt = "Table users has 5 rows. Query: SELECT COUNT(*) FROM users; What is the result?"
        result = 5
    elif level == 2:
        prompt = "Orders table has 8 rows where status='paid'. Query returns COUNT(status='paid'). What is it?"
        result = 8
    elif level == 3:
        prompt = (
            "users(id,name) = 3 rows; purchases(user_id,amount) = [ (1,10),(1,5),(2,7) ].\n"
            "Query: SELECT COUNT(*) FROM users u JOIN purchases p ON u.id=p.user_id;"
        )
        result = 3
    elif level == 4:
        prompt = (
            "sales(product, region, revenue): [('A','NA',5),('A','EU',3),('B','NA',4),('B','EU',7)].\n"
            "Query: SELECT region, SUM(revenue) r FROM sales GROUP BY region HAVING r > 7; How many rows?"
        )
        # NA:9 EU:10 -> both >7 => 2
        result = 2
    else:
        prompt = (
            "teams(id,name)=[(1,'red'),(2,'blue'),(3,'gold')]\n"
            "members(team_id,score)=[(1,5),(1,9),(2,4),(2,6),(2,10),(3,12)]\n"
            "Query: SELECT t.name, AVG(score) a FROM teams t\n"
            "JOIN members m ON t.id=m.team_id GROUP BY t.name HAVING a >= 8;"
        )
        # team1 avg 7, team2 avg 6.67, team3 avg 12 -> only gold
        result = 1

    distractors = _generate_numeric_distractors(result, rng)
    return _task_from_values(10, level, prompt, result, distractors, rng)


# =============================================================================
# FAMILY 11: html_css — specificity traps
# =============================================================================


def generate_html_css(level: int, rng: random.Random) -> TaskSpec:
    level = _normalize_difficulty(level)

    if level == 1:
        prompt = "Which rule wins? .btn {color: blue;} or .btn-primary {color: red;} on <button class='btn-primary'>"
        result = "red"
    elif level == 2:
        prompt = "#hero h1 {color: black;} .title {color: gray;} Which color for <h1 class='title' id='hero'>?"
        result = "black"
    elif level == 3:
        prompt = "div.note {color: green;} .note.warning {color: orange;} Which color for <div class='note warning'>?"
        result = "orange"
    elif level == 4:
        prompt = "p {color: blue;} .card p {color: purple;} .card > p.highlight {color: teal;} Which color for <div class='card'><p class='highlight'>"
        result = "teal"
    else:
        prompt = "#nav li.active a {color: crimson;} ul#nav li a:hover {color: navy;} .active > a {color: lime;} Active link hover color?"
        result = "crimson"

    distractors = [c for c in ["blue", "gray", "orange", "navy", "lime"] if c != result]
    rng.shuffle(distractors)
    return _task_from_values(11, level, prompt, result, distractors[:3], rng)


# =============================================================================
# FAMILY 12: bash_syntax — commands and quoting
# =============================================================================


def generate_bash_syntax(level: int, rng: random.Random) -> TaskSpec:
    level = _normalize_difficulty(level)

    if level == 1:
        prompt = "What does `echo $((2+3))` print?"
        result = 5
    elif level == 2:
        prompt = "Command: `echo 'a b' | wc -w` outputs?"
        result = 1
    elif level == 3:
        prompt = "Command: `touch a{1..3}.txt`; how many files created?"
        result = 3
    elif level == 4:
        prompt = "What is printed? `msg=hi; echo \"$msg\" | sed 's/h/H/'`"
        result = "Hi"
    else:
        prompt = "What is output of `printf \"a b\\nc\" | grep -o \"^a.*\" | wc -l`?"
        # printf produces two lines; grep matches first line starting with a b; count lines ->1
        result = 1

    distractors = _generate_numeric_distractors(result, rng) if isinstance(result, (int, float)) else ["hi", "h", "2"]
    return _task_from_values(12, level, prompt, result, distractors, rng)


# =============================================================================
# FAMILY 13: regex — precedence traps
# =============================================================================


def generate_regex(level: int, rng: random.Random) -> TaskSpec:
    level = _normalize_difficulty(level)

    if level == 1:
        prompt = "Pattern: `a|b` on 'cat' — does it match?"
        result = True
    elif level == 2:
        prompt = "Pattern: `ab?` on 'abbb' — first match?"
        result = "ab"
    elif level == 3:
        prompt = "Pattern: `(ab)+?b` on 'ababb' — does it match?"
        result = True
    elif level == 4:
        prompt = "Pattern: `^a(?=.+b).*c$` on 'axbyc' — match?"
        result = True
    else:
        prompt = "How many matches for `(?<!a)b(?!c)` in 'ab bc bbc bacb'?"
        # tokens: "ab","bc","bbc","bacb" with spaces
        text = "ab bc bbc bacb"
        count = 0
        for idx, ch in enumerate(text):
            if ch != "b":
                continue
            prev = text[idx - 1] if idx > 0 else ""
            nxt = text[idx + 1] if idx + 1 < len(text) else ""
            if prev == "a":
                continue
            if nxt == "c":
                continue
            count += 1
        result = count

    distractors = _generate_numeric_distractors(result, rng) if isinstance(result, (int, float)) else [not result, "no", "maybe"]
    return _task_from_values(13, level, prompt, result, distractors, rng)


# =============================================================================
# FAMILY 14: math_algebra — algebraic reasoning
# =============================================================================


def generate_math_algebra(level: int, rng: random.Random) -> TaskSpec:
    level = _normalize_difficulty(level)
    variant = rng.randint(0, 2)

    if level == 1:
        if variant == 0:
            a = rng.randint(1, 5)
            b = rng.randint(1, 5)
            prompt = f"Solve: x = {a} + {b}"
            result = a + b
        else:
            a = rng.randint(2, 6)
            prompt = f"Solve: x - {a} = {a}"
            result = a + a
    elif level == 2:
        if variant == 0:
            a = rng.randint(2, 6)
            b = rng.randint(1, 4)
            prompt = f"Solve: 2x + {b} = {2*a + b}"
            result = a
        else:
            a = rng.randint(2, 6)
            b = rng.randint(1, 4)
            prompt = f"Solve: 3x - {b} = {3*a - b}"
            result = a
    elif level == 3:
        a = rng.randint(2, 4)
        b = rng.randint(1, 3)
        prompt = f"Solve: x^2 - {b}x - {a*b} = 0 (positive root)"
        result = b + a
    elif level == 4:
        a = rng.randint(2, 5)
        b = rng.randint(3, 7)
        prompt = f"Solve for x: (x - {a})/{b} + (x + {a})/({b+1}) = 2"
        # (x-a)/b + (x+a)/(b+1)=2 => (x-a)(b+1)+(x+a)b = 2b(b+1)
        numerator = (b + 1) + b
        constant = -a * (b + 1) + a * b
        rhs = 2 * b * (b + 1)
        result = (rhs - constant) / numerator
    else:
        prompt = "Let f(x)=x^2-3x+2. Compute f(5) - f(2) + f(1)."
        def f(x): return x * x - 3 * x + 2
        result = f(5) - f(2) + f(1)

    distractors = _generate_numeric_distractors(result, rng)
    return _task_from_values(14, level, prompt, result, distractors, rng)


# =============================================================================
# FAMILY 15: physics_kinematics — basic physics equations
# =============================================================================


def generate_physics(level: int, rng: random.Random) -> TaskSpec:
    level = _normalize_difficulty(level)
    variant = rng.randint(0, 1)

    if level == 1:
        if variant == 0:
            u = rng.randint(2, 8)
            t = rng.randint(2, 6)
            prompt = f"Constant velocity motion: v={u} m/s, t={t}s. Displacement?"
            result = u * t
        else:
            d = rng.randint(10, 30)
            t = rng.randint(2, 5)
            prompt = f"Distance={d}m in {t}s at constant speed. What is speed?"
            result = d / t
    elif level == 2:
        u = rng.randint(2, 6)
        a = rng.randint(1, 4)
        t = rng.randint(2, 6)
        prompt = f"Use s = ut + 0.5at^2 with u={u}, a={a}, t={t}. Find s."
        result = u * t + 0.5 * a * t * t
    elif level == 3:
        m = rng.randint(2, 6)
        v = rng.randint(5, 15)
        prompt = f"Kinetic energy: m={m} kg, v={v} m/s. KE?"
        result = 0.5 * m * v * v
    elif level == 4:
        u = rng.randint(4, 10)
        a = -rng.randint(1, 3)
        t = rng.randint(3, 6)
        prompt = f"Deceleration: u={u} m/s, a={a} m/s^2 for t={t}s. Final velocity v = ?"
        result = u + a * t
    else:
        u = rng.randint(6, 12)
        angle = rng.choice([30, 45, 60])
        t = rng.randint(2, 4)
        g = 9.8
        vy = u * 0.5 if angle == 30 else (u / 2**0.5 if angle == 45 else u * 0.866)
        displacement = vy * t - 0.5 * g * t * t
        prompt = f"Projectile vertical: u={u} m/s at {angle}°, g=9.8, t={t}s. Vertical displacement?"
        result = round(displacement, 2)

    distractors = _generate_numeric_distractors(result, rng)
    return _task_from_values(15, level, prompt, result, distractors, rng)


# =============================================================================
# FAMILY 16: chemistry — stoichiometry & reactions
# =============================================================================


def generate_chemistry(level: int, rng: random.Random) -> TaskSpec:
    level = _normalize_difficulty(level)
    variant = rng.randint(0, 1)

    if level == 1:
        if variant == 0:
            prompt = "How many H atoms in H2O?"
            result = 2
        else:
            prompt = "How many O atoms in CO2?"
            result = 2
    elif level == 2:
        if variant == 0:
            prompt = "Balance: H2 + O2 -> H2O. Mole ratio H2:O2?"
            result = 2
        else:
            prompt = "Balance: N2 + 3H2 -> 2NH3. Mole ratio H2:N2?"
            result = 3
    elif level == 3:
        if variant == 0:
            prompt = "1 mol CaCO3 decomposes -> CaO + CO2. How many moles CO2?"
            result = 1
        else:
            prompt = "2KClO3 -> 2KCl + 3O2. Moles O2 from 2 mol KClO3?"
            result = 3
    elif level == 4:
        if variant == 0:
            prompt = "2Al + 3Cl2 -> 2AlCl3. If 4 mol Al react fully, moles of AlCl3?"
            result = 4
        else:
            prompt = "C + O2 -> CO2. If 5 mol O2 available with excess C, moles CO2?"
            result = 5
    else:
        if variant == 0:
            prompt = "C6H12O6 + 6O2 -> 6CO2 + 6H2O. Moles of O2 needed for 2 mol glucose?"
            result = 12
        else:
            prompt = "Fe2O3 + 3CO -> 2Fe + 3CO2. If 3 mol Fe2O3 react, moles Fe?"
            result = 6

    distractors = _generate_numeric_distractors(result, rng)
    return _task_from_values(16, level, prompt, result, distractors, rng)


# =============================================================================
# FAMILY 17: probability — conditional probability & combinatorics
# =============================================================================


def generate_probability(level: int, rng: random.Random) -> TaskSpec:
    level = _normalize_difficulty(level)
    variant = rng.randint(0, 1)

    if level == 1:
        if variant == 0:
            prompt = "A fair coin is flipped. P(heads)?"
            result = 0.5
        else:
            prompt = "Roll 1d4. P(rolling a 1)?"
            result = 0.25
    elif level == 2:
        if variant == 0:
            prompt = "Roll 1d6. P(rolling an even number)?"
            result = 0.5
        else:
            prompt = "Pick a card from a standard deck. P(card is heart)?"
            result = 13 / 52
    elif level == 3:
        if variant == 0:
            prompt = "Two fair coins. P(exactly one head)?"
            result = 0.5
        else:
            prompt = "Roll 1d6 twice. P(sum equals 7)?"
            result = 6 / 36
    elif level == 4:
        if variant == 0:
            prompt = "Deck 52 cards. Draw 2 without replacement. P(both hearts)?"
            result = (13 / 52) * (12 / 51)
        else:
            prompt = "Bag: 3 red, 5 blue. Draw 2 without replacement. P(no red)?"
            result = (5 / 8) * (4 / 7)
    else:
        prompt = "Bag: 2R,3B balls. Draw 2 without replacement. P(first red | second red)?"
        # P(first red and second red)/P(second red)
        p_rr = (2 / 5) * (1 / 4)
        p_sr = (2 / 5) * (1 / 4) + (3 / 5) * (2 / 4)
        result = p_rr / p_sr

    distractors = _generate_numeric_distractors(result, rng)
    return _task_from_values(17, level, prompt, round(result, 4), [round(d, 4) for d in distractors], rng)


# =============================================================================
# MAIN API
# =============================================================================

GENERATORS: List[Callable[[int, random.Random], TaskSpec]] = [
    generate_var_trace,
    generate_if_cond,
    generate_loop_count,
    generate_list_index,
    generate_bool_logic,
    generate_python_syntax,
    generate_java_syntax,
    generate_js_syntax,
    generate_cpp_syntax,
    generate_rust_syntax,
    generate_sql_syntax,
    generate_html_css,
    generate_bash_syntax,
    generate_regex,
    generate_math_algebra,
    generate_physics,
    generate_chemistry,
    generate_probability,
]


def generate_task(family_id: int, difficulty_id: int, seed: int | None = None) -> TaskSpec:
    """Generate a single task."""
    if not (0 <= family_id < NUM_FAMILIES):
        raise ValueError(f"family_id must be 0-{NUM_FAMILIES-1}")
    difficulty_id = _normalize_difficulty(difficulty_id)
    rng = random.Random(seed)
    return GENERATORS[family_id](difficulty_id, rng)


def generate_task_by_arm(arm_id: int, seed: int | None = None) -> TaskSpec:
    """arm_id = family_id * NUM_DIFFICULTIES + (difficulty_id-1)"""
    if not (0 <= arm_id < NUM_FAMILIES * NUM_DIFFICULTIES):
        raise ValueError(f"arm_id must be 0-{NUM_FAMILIES * NUM_DIFFICULTIES - 1}")

    family_id = arm_id // NUM_DIFFICULTIES
    difficulty_id = (arm_id % NUM_DIFFICULTIES) + 1  # store as 1–5
    return generate_task(family_id, difficulty_id, seed)


def generate_eval_dataset(tasks_per_type: int = 5, seed: int = 42) -> List[TaskSpec]:
    """Generate a fixed evaluation dataset."""
    rng = random.Random(seed)
    dataset: List[TaskSpec] = []

    for family_id in range(NUM_FAMILIES):
        for difficulty_id in range(1, NUM_DIFFICULTIES + 1):
            for _ in range(tasks_per_type):
                task_seed = rng.randint(0, 2**31 - 1)
                dataset.append(generate_task(family_id, difficulty_id, task_seed))

    return dataset


def iter_eval_dataset(tasks_per_type: int = 5, seed: int = 42):
    """
    Memory-light iterator over the eval dataset.
    
    Useful when you want to stream tasks instead of materialising the full list.
    """
    rng = random.Random(seed)
    for family_id in range(NUM_FAMILIES):
        for difficulty_id in range(1, NUM_DIFFICULTIES + 1):
            for _ in range(tasks_per_type):
                task_seed = rng.randint(0, 2**31 - 1)
                yield generate_task(family_id, difficulty_id, task_seed)


def arm_to_name(arm_id: int) -> str:
    family_id = arm_id // NUM_DIFFICULTIES
    difficulty_id = (arm_id % NUM_DIFFICULTIES) + 1
    return f"{FAMILY_NAMES[family_id]}_lvl{difficulty_id}"


if __name__ == "__main__":
    # Quick manual smoke test
    rng = random.Random(123)
    for fam in range(3):
        for diff in range(1, 6):
            task = generate_task(fam, diff, seed=rng.randint(0, 9999))
            print(f"[{FAMILY_NAMES[fam]}][{DIFFICULTY_NAMES[diff-1]}]")
            print(task.human_prompt)
            print(task.human_choices)
            print("Correct:", task.human_choices[task.correct_action])
            print("-" * 40)
