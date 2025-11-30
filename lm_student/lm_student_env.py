"""Text-based student environment using GPT-2 tokenization."""

import random
import string
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.utils import seeding
from transformers import PreTrainedTokenizerBase


def _random_string(length: int, mixed_case: bool = False) -> str:
    letters = string.ascii_letters if mixed_case else string.ascii_lowercase
    return "".join(random.choice(letters) for _ in range(length))


def _build_task_text(task_id: int, difficulty: int) -> Tuple[str, List[str], int]:
    """Return (prompt_text, choices, correct_idx) for textual tasks 0-4."""
    if task_id == 0:
        length_map = {0: 3, 1: 5, 2: 10}
        length = length_map[difficulty]
        source = _random_string(length)
        correct = source[::-1]
        choices = [
            correct,
            "".join(random.sample(source, len(source))),
            source,
            _random_string(length),
        ]
        choices = choices[:4]
        correct_idx = 0
        prompt = f"Reverse the string: {source}"
        return prompt, choices, correct_idx
    if task_id == 1:
        ranges = {0: (0, 10), 1: (0, 100), 2: (0, 500)}
        low, high = ranges[difficulty]
        a, b = random.randint(low, high), random.randint(low, high)
        correct = a + b
        choices = [
            correct,
            correct + 1,
            max(0, correct - 1),
            correct + random.choice([-3, 3]),
        ]
        choices = [str(c) for c in choices]
        correct_idx = 0
        prompt = f"What is {a} + {b}?"
        return prompt, choices, correct_idx
    if task_id == 2:
        length_map = {0: (3, 5), 1: (6, 12), 2: (10, 20)}
        lo, hi = length_map[difficulty]
        s = _random_string(random.randint(lo, hi), mixed_case=(difficulty == 2))
        vowels = set("aeiouAEIOU")
        correct = sum(1 for ch in s if ch in vowels)
        candidates = {correct, correct + 1, max(0, correct - 1)}
        while len(candidates) < 4:
            candidates.add(max(0, correct + random.randint(-3, 3)))
        choices = [str(c) for c in list(candidates)[:4]]
        random.shuffle(choices)
        correct_idx = choices.index(str(correct))
        prompt = f"How many vowels are in '{s}'?"
        return prompt, choices, correct_idx
    if task_id == 3:
        ranges = {0: (0, 10), 1: (0, 50), 2: (0, 200)}
        low, high = ranges[difficulty]
        a, b = random.randint(low, high), random.randint(low, high)
        correct = a - b
        candidates = {correct, correct + 1, correct - 1, correct + 5, correct - 5}
        choices = [str(c) for c in list(candidates)[:4]]
        random.shuffle(choices)
        correct_idx = choices.index(str(correct))
        prompt = f"What is {a} - {b}?"
        return prompt, choices, correct_idx
    if task_id == 4:
        length_map = {0: (3, 5), 1: (5, 10), 2: (10, 20)}
        lo, hi = length_map[difficulty]
        make_pal = random.random() < 0.5
        if make_pal:
            half = _random_string(random.randint(lo, hi) // 2, mixed_case=(difficulty == 2))
            middle = _random_string(1, mixed_case=(difficulty == 2))
            s = half + middle + half[::-1]
        else:
            s = _random_string(random.randint(lo, hi), mixed_case=(difficulty == 2))
        is_pal = s == s[::-1]
        choices = ["true", "false", "maybe", "undefined"]
        correct_idx = choices.index("true" if is_pal else "false")
        prompt = f"Is '{s}' a palindrome?"
        return prompt, choices, correct_idx
    raise ValueError(f"Unsupported task_id {task_id}")


class LMStudentEnv(gym.Env):
    """Single-step environment emitting tokenized text prompts for PPO."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        task_id: int = 0,
        difficulty: int = 0,
        max_length: int = 64,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.task_id = task_id
        self.difficulty = difficulty
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.np_random, _ = seeding.np_random(seed)

        vocab_size = getattr(tokenizer, "vocab_size", 50257)
        self.observation_space = spaces.Box(
            low=0,
            high=vocab_size,
            shape=(max_length,),
            dtype=np.int64,
        )
        self.action_space = spaces.Discrete(4)

        self.correct_idx: int = 0
        self.choices: List[str] = []

    def _tokenize(self, text: str) -> np.ndarray:
        encoded = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="np",
        )
        return encoded["input_ids"].astype(np.int64).squeeze(0)

    def _sample_task(self) -> np.ndarray:
        prompt, choices, correct_idx = _build_task_text(self.task_id, self.difficulty)
        self.choices = choices
        self.correct_idx = correct_idx
        obs = self._tokenize(self._format_prompt(prompt, choices))
        return obs

    @staticmethod
    def _format_prompt(prompt: str, choices: List[str]) -> str:
        lines = [f"Task: {prompt}", "Choices:"]
        labels = ["A", "B", "C", "D"]
        for lbl, choice in zip(labels, choices):
            lines.append(f"{lbl}) {choice}")
        return "\n".join(lines)

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            self.np_random, _ = seeding.np_random(seed)
        obs = self._sample_task()
        return obs, {"choices": self.choices, "correct_idx": self.correct_idx}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        reward = 1.0 if int(action) == self.correct_idx else 0.0
        obs = self._sample_task()
        info = {"choices": self.choices, "correct_idx": self.correct_idx}
        return obs, reward, True, False, info
