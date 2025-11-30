"""Medium-complexity task generator.

Balanced between the tiny mock generator and the full production generator:
- STEM-heavy topics (minimal world knowledge needed)
- Five difficulty tiers mapped to reasoning steps (1–5)
- Purely templated MCQs for speed and LM-light behavior
"""

import random
from typing import List
from interfaces import Task, TaskGeneratorInterface


class MediumTaskGenerator(TaskGeneratorInterface):
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.topics = [
            "var_trace",
            "if_cond",
            "loop_count",
            "list_index",
            "bool_logic",
            "math_chain",
            "probability",
            "physics",
            "chemistry",
            "programming",
        ]
        self.difficulties = [
            "easy",
            "medium",
            "hard",
            "expert",
            "master",
        ]
        self.task_counter = 0

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _level(self, difficulty: str) -> int:
        if difficulty not in self.difficulties:
            raise ValueError(f"Unknown difficulty {difficulty}")
        return self.difficulties.index(difficulty) + 1  # 1-5

    def _choices(self, correct):
        """Build 4 choices with small perturbations."""
        distractors = set()
        while len(distractors) < 3:
            if isinstance(correct, (int, float)):
                delta = self.rng.randint(-3, 3) or 1
                distractors.add(correct + delta)
            else:
                distractors.add(f"{correct}?!")
        distractors = list(distractors)[:3]
        options = [correct] + distractors
        self.rng.shuffle(options)
        answer = options.index(correct)
        return [str(o) for o in options], answer

    def _task(self, topic: str, difficulty: str, prompt: str, correct) -> Task:
        choices, answer = self._choices(correct)
        self.task_counter += 1
        return Task(
            passage=prompt,
            question="What is the result?",
            choices=choices,
            answer=answer,
            topic=topic,
            difficulty=difficulty,
            task_id=f"{topic}_{difficulty}_{self.task_counter}",
        )

    # ------------------------------------------------------------------ #
    # Topic generators
    # ------------------------------------------------------------------ #
    def _gen_var_trace(self, level: int, difficulty: str) -> Task:
        a = self.rng.randint(2, 9)
        b = self.rng.randint(2, 9)
        c = self.rng.randint(1, 4)
        if level == 1:
            result = a + b
            prompt = f"a = {a}\nb = {b}\na = a + b\nWhat is a?"
        elif level == 2:
            result = (a + b) * c
            prompt = f"a = {a}\nb = {b}\nc = {c}\na = (a + b) * c\nWhat is a?"
        elif level == 3:
            d = self.rng.randint(1, 4)
            result = (a + b) * c - d
            prompt = f"a = {a}\nb = {b}\nc = {c}\nd = {d}\na = (a + b) * c - d\nWhat is a?"
        elif level == 4:
            d = self.rng.randint(1, 4)
            e = self.rng.randint(1, 3)
            result = ((a + b) * c - d) + (a % e)
            prompt = (
                f"a = {a}\nb = {b}\nc = {c}\nd = {d}\ne = {e}\n"
                "a = (a + b) * c - d + (a % e)\nWhat is a?"
            )
        else:
            arr = [self.rng.randint(1, 5) for _ in range(3)]
            result = sum(arr) + (a * b) - c
            prompt = (
                f"a = {a}\nb = {b}\nc = {c}\narr = {arr}\n"
                "a = sum(arr) + (a * b) - c\nWhat is a?"
            )
        return self._task("var_trace", difficulty, prompt, result)

    def _gen_if_cond(self, level: int, difficulty: str) -> Task:
        x = self.rng.randint(1, 5)
        y = self.rng.randint(1, 5)
        z = self.rng.randint(1, 5)
        if level == 1:
            result = x > y
            prompt = f"x={x}, y={y}\nExpression: x > y"
        elif level == 2:
            result = (x > y) and (y > z)
            prompt = f"x={x}, y={y}, z={z}\n(x > y) and (y > z)"
        elif level == 3:
            result = (x + y > z) or (z % 2 == 0)
            prompt = f"x={x}, y={y}, z={z}\n(x + y > z) or (z % 2 == 0)"
        elif level == 4:
            result = (x == y) ^ (z > x + y)
            prompt = f"x={x}, y={y}, z={z}\n(x == y) xor (z > x + y)"
        else:
            result = ((x + z) % 2 == 0) and not (y % 2)
            prompt = f"x={x}, y={y}, z={z}\n((x + z) % 2 == 0) and not (y % 2)"
        return self._task("if_cond", difficulty, prompt, result)

    def _gen_loop_count(self, level: int, difficulty: str) -> Task:
        n = self.rng.randint(2, 6)
        if level == 1:
            result = n
            prompt = f"count=0\nfor i in range({n}):\n    count += 1\nWhat is count?"
        elif level == 2:
            result = sum(range(n))
            prompt = f"acc=0\nfor i in range({n}):\n    acc += i\nWhat is acc?"
        elif level == 3:
            step = self.rng.randint(2, 3)
            rng = list(range(0, n * step, step))
            result = len(rng)
            prompt = f"count=0\nfor i in range(0, {n*step}, {step}):\n    count += 1\nWhat is count?"
        elif level == 4:
            result = sum(i for i in range(n) if i % 2 == 0)
            prompt = f"acc=0\nfor i in range({n}):\n    if i % 2 == 0:\n        acc += i\nWhat is acc?"
        else:
            result = 0
            for i in range(n):
                for j in range(i):
                    result += j
            prompt = (
                f"acc=0\nfor i in range({n}):\n"
                "    for j in range(i):\n"
                "        acc += j\nWhat is acc?"
            )
        return self._task("loop_count", difficulty, prompt, result)

    def _gen_list_index(self, level: int, difficulty: str) -> Task:
        arr = [self.rng.randint(1, 9) for _ in range(5)]
        if level == 1:
            result = arr[0]
            prompt = f"arr = {arr}\nWhat is arr[0]?"
        elif level == 2:
            result = arr[-1]
            prompt = f"arr = {arr}\nWhat is arr[-1]?"
        elif level == 3:
            result = sum(arr[:3])
            prompt = f"arr = {arr}\nWhat is sum(arr[:3])?"
        elif level == 4:
            result = arr[1] + arr[-2]
            prompt = f"arr = {arr}\nWhat is arr[1] + arr[-2]?"
        else:
            result = sum(arr[1:-1])
            prompt = f"arr = {arr}\nWhat is sum(arr[1:-1])?"
        return self._task("list_index", difficulty, prompt, result)

    def _gen_bool_logic(self, level: int, difficulty: str) -> Task:
        flags = [self.rng.choice([True, False]) for _ in range(3)]
        a, b, c = flags
        if level == 1:
            result = a and b
            prompt = f"a={a}, b={b}\nExpression: a and b"
        elif level == 2:
            result = (a or b) and not c
            prompt = f"a={a}, b={b}, c={c}\n(a or b) and not c"
        elif level == 3:
            result = (a ^ b) or (b and c)
            prompt = f"a={a}, b={b}, c={c}\n(a xor b) or (b and c)"
        elif level == 4:
            result = (a == b) and (b != c)
            prompt = f"a={a}, b={b}, c={c}\n(a == b) and (b != c)"
        else:
            result = (a or (b and c)) ^ (not a and b)
            prompt = f"a={a}, b={b}, c={c}\n(a or (b and c)) xor (not a and b)"
        return self._task("bool_logic", difficulty, prompt, result)

    def _gen_math_chain(self, level: int, difficulty: str) -> Task:
        x = self.rng.randint(1, 8)
        y = self.rng.randint(1, 8)
        z = self.rng.randint(1, 8)
        if level == 1:
            result = x + y
            prompt = f"x={x}, y={y}\nCompute x + y"
        elif level == 2:
            result = (x + y) * z
            prompt = f"x={x}, y={y}, z={z}\nCompute (x + y) * z"
        elif level == 3:
            result = (x + y) * z - x
            prompt = f"x={x}, y={y}, z={z}\nCompute (x + y) * z - x"
        elif level == 4:
            result = (x * y) + (z ** 2)
            prompt = f"x={x}, y={y}, z={z}\nCompute x*y + z^2"
        else:
            result = (x + y + z) * (x - y + z)
            prompt = f"x={x}, y={y}, z={z}\nCompute (x+y+z)*(x - y + z)"
        return self._task("math_chain", difficulty, prompt, result)

    def _gen_probability(self, level: int, difficulty: str) -> Task:
        if level == 1:
            sides = 6
            result = 1 / sides
            prompt = "Roll a fair 6-sided die. P(result = 3)?"
        elif level == 2:
            result = 2 / 6
            prompt = "Roll a fair 6-sided die. P(result is even)?"
        elif level == 3:
            result = 1 - (5 / 6) * (5 / 6)
            prompt = "Roll a fair die twice. P(at least one 6)?"
        elif level == 4:
            result = (2 / 6) * (1 / 2)
            prompt = "Pick a die result (even) then flip a fair coin (heads). P(both)?"
        else:
            result = (3 / 6) * (2 / 6)
            prompt = "Roll two dice. P(first <=3 and second even)?"
        return self._task("probability", difficulty, prompt, round(result, 3))

    def _gen_physics(self, level: int, difficulty: str) -> Task:
        v = self.rng.randint(2, 6)
        t = self.rng.randint(2, 5)
        a = self.rng.randint(1, 3)
        if level == 1:
            result = v * t
            prompt = f"Velocity={v} m/s, time={t}s. Distance?"
        elif level == 2:
            result = v * t + 0.5 * a * t * t
            prompt = f"v0={v}, a={a}, t={t}. Distance with constant acceleration?"
        elif level == 3:
            result = v + a * t
            prompt = f"v0={v}, a={a}, t={t}. Final velocity?"
        elif level == 4:
            result = (v * t) + (a * t)
            prompt = f"v0={v}, a={a}, t={t}. Displacement using v*t + a*t?"
        else:
            result = (v + a * t) * t
            prompt = f"v0={v}, a={a}, t={t}. Approx distance using v_final * t?"
        return self._task("physics", difficulty, prompt, round(result, 2))

    def _gen_chemistry(self, level: int, difficulty: str) -> Task:
        # Simple ratio / balancing style
        h2 = self.rng.randint(1, 4)
        o2 = self.rng.randint(1, 3)
        if level == 1:
            result = h2 * 2
            prompt = f"{h2} molecules of H2 -> How many H atoms total?"
        elif level == 2:
            result = o2 * 2
            prompt = f"{o2} molecules of O2 -> How many O atoms total?"
        elif level == 3:
            result = min(h2 // 2, o2) * 2
            prompt = f"{h2} H2 and {o2} O2 combine to form H2O. How many H2O max?"
        elif level == 4:
            result = min(h2 // 2, o2) * 2 * 18  # grams, 18g/mol per H2O
            prompt = f"{h2} H2 and {o2} O2 -> H2O. Mass of water formed (grams, 18g/unit)?"
        else:
            result = min(h2 // 2, o2)
            prompt = f"{h2} H2 and {o2} O2 -> limiting reagent? Count of limiting pairs."
        return self._task("chemistry", difficulty, prompt, result)

    def _gen_programming(self, level: int, difficulty: str) -> Task:
        lst = [self.rng.randint(1, 4) for _ in range(4)]
        if level == 1:
            result = len(lst)
            prompt = f"lst = {lst}\nlen(lst)?"
        elif level == 2:
            result = lst[0] + lst[-1]
            prompt = f"lst = {lst}\nlst[0] + lst[-1]?"
        elif level == 3:
            result = sum(x for x in lst if x % 2 == 0)
            prompt = f"lst = {lst}\nsum of even elements?"
        elif level == 4:
            result = [x * 2 for x in lst if x % 2][0] if any(x % 2 for x in lst) else 0
            prompt = f"lst = {lst}\nfirst odd doubled using list comp?"
        else:
            result = sum(lst[i] * (i + 1) for i in range(len(lst)))
            prompt = f"lst = {lst}\nsum(lst[i]*(i+1))?"
        return self._task("programming", difficulty, prompt, result)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def generate_task(self, topic: str, difficulty: str) -> Task:
        if topic not in self.topics:
            raise ValueError(f"Unknown topic: {topic}")
        level = self._level(difficulty)

        generators = {
            "var_trace": self._gen_var_trace,
            "if_cond": self._gen_if_cond,
            "loop_count": self._gen_loop_count,
            "list_index": self._gen_list_index,
            "bool_logic": self._gen_bool_logic,
            "math_chain": self._gen_math_chain,
            "probability": self._gen_probability,
            "physics": self._gen_physics,
            "chemistry": self._gen_chemistry,
            "programming": self._gen_programming,
        }
        return generators[topic](level, difficulty)

    def get_available_topics(self) -> List[str]:
        return list(self.topics)

    def get_available_difficulties(self) -> List[str]:
        return list(self.difficulties)
