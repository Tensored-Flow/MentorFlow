"""Student package for TeachRL.

Exports:
- PPO-based RL student: `StudentAgent` (see `ppo_agent.py`)
- Small-language-model student: `run_student_step` (see `slm_agent.py`)

Note: The PPO-based student depends on optional libraries (e.g. numpy,
stable-baselines3). If those are not installed, importing ``StudentAgent``
will gracefully fail, but ``run_student_step`` remains available.
"""

try:  # Optional: RL student requires extra dependencies
    from .ppo_agent import StudentAgent  # type: ignore
except Exception:  # ImportError, ModuleNotFoundError, etc.
    StudentAgent = None  # type: ignore

from .slm_agent import run_student_step

