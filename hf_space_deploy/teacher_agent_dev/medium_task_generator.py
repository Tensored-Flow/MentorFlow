"""Medium-complexity task generator for HF Space.

STEM-heavy, LM-light, multiple templates per topic/difficulty.
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
        self.difficulties = ["easy", "medium", "hard", "expert", "master"]
        self.task_counter = 0

    def _level(self, difficulty: str) -> int:
        if difficulty not in self.difficulties:
            raise ValueError(f"Unknown difficulty {difficulty}")
        return self.difficulties.index(difficulty) + 1

    def _choices(self, correct):
        distractors = set()
        while len(distractors) < 3:
            if isinstance(correct, (int, float)):
                delta = self.rng.randint(-3, 3) or 1
                distractors.add(correct + delta)
            else:
                distractors.add(f"{correct}?!")
        options = [correct] + list(distractors)[:3]
        self.rng.shuffle(options)
        return [str(o) for o in options], options.index(correct)

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

    def _gen_var_trace(self, level: int, difficulty: str) -> Task:
        variant = self.rng.randint(0, 2)
        if level == 1:
            if variant == 0:
                a = self.rng.randint(2, 9)
                b = self.rng.randint(2, 9)
                result = a + b
                prompt = f"a = {a}\\nb = {b}\\na = a + b\\nWhat is a?"
            elif variant == 1:
                a = self.rng.randint(3, 7)
                b = self.rng.randint(1, 4)
                result = a + b - 1
                prompt = f"a = {a}\\na += {b}\\na -= 1\\nWhat is a?"
            else:
                a = self.rng.randint(2, 6)
                b = self.rng.randint(1, 3)
                result = a * 2 - b
                prompt = f"a = {a}\\na = a * 2 - {b}\\nWhat is a?"
        elif level == 2:
            if variant == 0:
                x = self.rng.randint(3, 8)
                y = x + self.rng.randint(2, 5)
                z = y - self.rng.randint(1, 3)
                result = (y - (y - z)) * 2 - x
                prompt = f"x = {x}\\ny = x + {y - x}\\nz = y - {y - z}\\nw = z * 2 - x\\nWhat is w?"
            elif variant == 1:
                x = self.rng.randint(2, 6)
                y = self.rng.randint(2, 6)
                z = self.rng.randint(1, 4)
                result = (x + y) + (x + y - z)
                prompt = (
                    f"x = {x}\\ny = {y}\\nfor i in range(2):\\n    x = x + y\\n    y = y - {z}\\nWhat is x?"
                )
            else:
                base = self.rng.randint(2, 6)
                inc = self.rng.randint(1, 3)
                dec = self.rng.randint(1, 2)
                result = ((base + inc) * 2) - dec
                prompt = f"val = {base}\\nval += {inc}\\nval *= 2\\nval -= {dec}\\nWhat is val?"
        elif level == 3:
            if variant == 0:
                base = self.rng.randint(2, 6)
                inc = self.rng.randint(2, 4)
                dec = self.rng.randint(1, 3)
                a = base + inc
                b = (a - dec) * 2
                c = b + (a % dec)
                result = c
                prompt = f"a = {base} + {inc}\\nb = (a - {dec}) * 2\\nc = b + (a % {dec})\\nWhat is c?"
            elif variant == 1:
                vals = [self.rng.randint(1, 5) for _ in range(3)]
                a, b, c = vals
                a = a + b
                b = b + c
                c = a + b
                result = c
                prompt = (
                    f"a, b, c = {vals}\\na = a + b\\nb = b + c\\nc = a + b\\nWhat is c?"
                )
            else:
                nums = [self.rng.randint(1, 4) for _ in range(4)]
                result = sum(v * (i + 1) for i, v in enumerate(nums))
                prompt = (
                    f"nums = {nums}\\ntotal = 0\\nfor i, v in enumerate(nums):\\n    total += v * (i+1)\\nWhat is total?"
                )
        elif level == 4:
            seq = [self.rng.randint(1, 9) for _ in range(4)]
            acc = 0
            for i, val in enumerate(seq):
                if i % 2 == 0:
                    acc += val * (i + 1)
                else:
                    acc -= val // (i + 1)
            acc += seq[0]
            result = acc
            prompt = (
                f"seq = {seq}\\nacc = 0\\nfor i, v in enumerate(seq):\\n    if i % 2 == 0: acc += v * (i+1)\\n    else: acc -= v // (i+1)\\nacc += seq[0]\\nWhat is acc?"
            )
        else:
            grid = [[self.rng.randint(1, 5) for _ in range(3)] for _ in range(2)]
            carry = self.rng.randint(1, 3)
            for i, row in enumerate(grid):
                for j, val in enumerate(row):
                    if (i + j) % 2 == 0:
                        carry += val + j
                    else:
                        carry -= max(1, val - i)
                grid[i][0] += carry % 3
            result = carry + grid[1][0] - grid[0][-1]
            prompt = (
                f"grid = {grid}\\ncarry = {carry % 5}\\nfor i, row in enumerate(grid):\\n    for j, v in enumerate(row):\\n        if (i + j) % 2 == 0: carry += v + j\\n        else: carry -= max(1, v - i)\\n    grid[i][0] += carry % 3\\nresult = carry + grid[1][0] - grid[0][-1]\\nWhat is result?"
            )
        return self._task("var_trace", difficulty, prompt, result)

    def _gen_if_cond(self, level: int, difficulty: str) -> Task:
        variant = self.rng.randint(0, 2)
        if level == 1:
            if variant == 0:
                x = self.rng.randint(1, 9)
                y = x + self.rng.randint(1, 4)
                result = x < y
                prompt = f"x = {x}\\ny = {y}\\nIs x < y?"
            else:
                a = self.rng.randint(1, 5)
                b = self.rng.randint(1, 5)
                result = (a * 2) >= (b + 1)
                prompt = f"a = {a}\\nb = {b}\\nIs (a * 2) >= (b + 1)?"
        elif level == 2:
            if variant == 0:
                a = self.rng.randint(2, 6)
                b = self.rng.randint(1, 4)
                c = self.rng.randint(3, 9)
                result = (a + b > c) and (c % 2 == 1)
                prompt = f"a = {a}\\nb = {b}\\nc = {c}\\nIs (a + b > c) and (c % 2 == 1)?"
            else:
                x = self.rng.randint(2, 6)
                y = self.rng.randint(2, 6)
                z = self.rng.randint(1, 4)
                result = (x - z < y) or (y % 2 == 0)
                prompt = f"x={x}, y={y}, z={z}\\nIs (x - z < y) or (y % 2 == 0)?"
        else:
            x = self.rng.randint(2, 6)
            y = self.rng.randint(2, 6)
            z = self.rng.randint(5, 12)
            result = (x == y) or (z < x + y) and not (x > z)
            prompt = f"x = {x}\\ny = {y}\\nz = {z}\\nEvaluate: (x == y) or (z < x + y) and not(x > z)"
        return self._task("if_cond", difficulty, prompt, result)

    def _gen_loop_count(self, level: int, difficulty: str) -> Task:
        variant = self.rng.randint(0, 1)
        if level == 1:
            if variant == 0:
                n = self.rng.randint(2, 5)
                result = n
                prompt = f"count = 0\\nfor i in range({n}):\\n    count += 1\\nWhat is count?"
            else:
                n = self.rng.randint(2, 5)
                start = self.rng.randint(0, 2)
                result = start + sum(range(n))
                prompt = f"count = {start}\\nfor i in range({n}):\\n    count += i\\nWhat is count?"
        elif level == 2:
            start = self.rng.randint(8, 14)
            step = self.rng.randint(2, 4)
            count = 0
            x = start
            while x > 0:
                x -= step
                count += 1
            result = count
            prompt = f"x = {start}\\nstep = {step}\\ncount = 0\\nwhile x > 0:\\n    x -= step\\n    count += 1\\nWhat is count?"
        else:
            outer = self.rng.randint(2, 4)
            inner = self.rng.randint(2, 4)
            count = 0
            for i in range(outer):
                for j in range(inner):
                    if (i + j) % 2 == 0:
                        count += 1
            result = count
            prompt = f"count = 0\\nfor i in range({outer}):\\n    for j in range({inner}):\\n        if (i + j) % 2 == 0:\\n            count += 1\\nWhat is count?"
        return self._task("loop_count", difficulty, prompt, result)

    def _gen_list_index(self, level: int, difficulty: str) -> Task:
        arr = [self.rng.randint(1, 9) for _ in range(5)]
        if level == 1:
            result = arr[0]
            prompt = f"arr = {arr}\\nWhat is arr[0]?"
        elif level == 2:
            result = arr[-1]
            prompt = f"arr = {arr}\\nWhat is arr[-1]?"
        else:
            result = sum(arr[1:-1])
            prompt = f"arr = {arr}\\nWhat is sum(arr[1:-1])?"
        return self._task("list_index", difficulty, prompt, result)

    def _gen_bool_logic(self, level: int, difficulty: str) -> Task:
        flags = [self.rng.choice([True, False]) for _ in range(3)]
        a, b, c = flags
        if level == 1:
            result = a and b
            prompt = f"a={a}, b={b}\\nExpression: a and b"
        elif level == 2:
            result = (a or b) and not c
            prompt = f"a={a}, b={b}, c={c}\\n(a or b) and not c"
        else:
            result = (a ^ b) or (b and c)
            prompt = f"a={a}, b={b}, c={c}\\n(a xor b) or (b and c)"
        return self._task("bool_logic", difficulty, prompt, result)

    def _gen_math_chain(self, level: int, difficulty: str) -> Task:
        x = self.rng.randint(1, 8)
        y = self.rng.randint(1, 8)
        z = self.rng.randint(1, 8)
        if level == 1:
            result = x + y
            prompt = f"x={x}, y={y}\\nCompute x + y"
        elif level == 2:
            result = (x + y) * z
            prompt = f"x={x}, y={y}, z={z}\\nCompute (x + y) * z"
        else:
            result = (x + y + z) * (x - y + z)
            prompt = f"x={x}, y={y}, z={z}\\nCompute (x+y+z)*(x - y + z)"
        return self._task("math_chain", difficulty, prompt, result)

    def _gen_probability(self, level: int, difficulty: str) -> Task:
        variant = self.rng.randint(0, 1)
        if level == 1:
            if variant == 0:
                result = 0.5
                prompt = "A fair coin is flipped. P(heads)?"
            else:
                result = 0.25
                prompt = "Roll 1d4. P(rolling a 1)?"
        elif level == 2:
            if variant == 0:
                result = 0.5
                prompt = "Roll 1d6. P(rolling an even number)?"
            else:
                result = 13 / 52
                prompt = "Pick a card from a standard deck. P(card is heart)?"
        else:
            result = 6 / 36
            prompt = "Roll 1d6 twice. P(sum equals 7)?"
        return self._task("probability", difficulty, prompt, round(result, 3))

    def _gen_physics(self, level: int, difficulty: str) -> Task:
        variant = self.rng.randint(0, 1)
        if level == 1:
            if variant == 0:
                u = self.rng.randint(2, 8)
                t = self.rng.randint(2, 6)
                result = u * t
                prompt = f"Constant velocity motion: v={u} m/s, t={t}s. Displacement?"
            else:
                d = self.rng.randint(10, 30)
                t = self.rng.randint(2, 5)
                result = d / t
                prompt = f"Distance={d}m in {t}s at constant speed. What is speed?"
        elif level == 2:
            u = self.rng.randint(2, 6)
            a = self.rng.randint(1, 4)
            t = self.rng.randint(2, 6)
            result = u * t + 0.5 * a * t * t
            prompt = f"Use s = ut + 0.5at^2 with u={u}, a={a}, t={t}. Find s."
        else:
            m = self.rng.randint(2, 6)
            v = self.rng.randint(5, 15)
            result = 0.5 * m * v * v
            prompt = f"Kinetic energy: m={m} kg, v={v} m/s. KE?"
        return self._task("physics", difficulty, prompt, round(result, 3))

    def _gen_chemistry(self, level: int, difficulty: str) -> Task:
        variant = self.rng.randint(0, 1)
        if level == 1:
            if variant == 0:
                prompt = "How many H atoms in H2O?"
                result = 2
            else:
                prompt = "How many O atoms in CO2?"
                result = 2
        elif level == 2:
            prompt = "Balance: H2 + O2 -> H2O. Mole ratio H2:O2?"
            result = 2
        else:
            prompt = "2Al + 3Cl2 -> 2AlCl3. If 4 mol Al react fully, moles of AlCl3?"
            result = 4
        return self._task("chemistry", difficulty, prompt, result)

    def _gen_programming(self, level: int, difficulty: str) -> Task:
        lst = [self.rng.randint(1, 4) for _ in range(4)]
        if level == 1:
            result = len(lst)
            prompt = f"lst = {lst}\\nlen(lst)?"
        elif level == 2:
            result = lst[0] + lst[-1]
            prompt = f"lst = {lst}\\nlst[0] + lst[-1]?"
        else:
            result = sum(x for x in lst if x % 2 == 0)
            prompt = f"lst = {lst}\\nsum of even elements?"
        return self._task("programming", difficulty, prompt, result)

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
