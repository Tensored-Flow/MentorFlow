"""Utilities for encoding observations and answer choices into fixed-length vectors.

An observation is represented as a 22-length vector:
    [task_id, difficulty, token_1, ..., token_20]

Tokens are encoded using:
- Letters: ASCII code via ord()
- Digits: integer value
- Symbols: predefined mapping
All remaining positions are padded with zeros to reach the target length.
"""

from typing import List

# Symbol mapping aligned with the task specification.
SYMBOL_MAP = {
    "+": 41,
    "-": 42,
    "*": 43,
    "/": 44,
    "=": 45,
}


def encode_string_to_vector(s: str, max_len: int = 20) -> List[int]:
    """Encode a string into a fixed-length list of integers.

    Args:
        s: Input string to encode.
        max_len: Maximum number of characters to encode.

    Returns:
        A list of length ``max_len`` containing encoded tokens.
    """
    tokens: List[int] = []
    for ch in s[:max_len]:
        if ch.isalpha():
            tokens.append(ord(ch))
        elif ch.isdigit():
            tokens.append(int(ch))
        elif ch in SYMBOL_MAP:
            tokens.append(SYMBOL_MAP[ch])
        else:
            tokens.append(0)

    # Pad with zeros to reach the desired length.
    while len(tokens) < max_len:
        tokens.append(0)

    return tokens[:max_len]


def build_observation(task_id: int, difficulty: int, prompt: str, max_len: int = 20) -> List[int]:
    """Construct the 22-length observation vector for the student environment."""
    encoded_prompt = encode_string_to_vector(prompt, max_len=max_len)
    return [task_id, difficulty, *encoded_prompt]
