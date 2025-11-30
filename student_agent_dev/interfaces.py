"""
Shared interfaces for all components.

DO NOT MODIFY - must match teacher and task generator teams.
"""

from dataclasses import dataclass
from typing import List, Dict
from abc import ABC, abstractmethod


@dataclass
class Task:
    """A reading comprehension task."""
    passage: str
    question: str
    choices: List[str]  # 4 choices: ['A) ...', 'B) ...', 'C) ...', 'D) ...']
    answer: int  # Index of correct answer (0-3)
    topic: str  # e.g., 'history', 'science', 'literature', 'geography', 'current_events'
    difficulty: str  # 'easy', 'medium', 'hard'
    task_id: str


@dataclass
class StudentState:
    """Student's current learning state."""
    topic_accuracies: Dict[str, float]  # topic -> accuracy (0.0-1.0)
    topic_attempts: Dict[str, int]  # topic -> number of attempts
    time_since_practice: Dict[str, float]  # topic -> time since last practice
    total_timesteps: int
    current_time: float


@dataclass
class TeacherAction:
    """Teacher's decision about what to teach next."""
    topic: str
    difficulty: str
    is_review: bool


class TaskGeneratorInterface(ABC):
    @abstractmethod
    def generate_task(self, topic: str, difficulty: str) -> Task:
        pass
    
    @abstractmethod
    def get_available_topics(self) -> List[str]:
        pass
    
    @abstractmethod
    def get_available_difficulties(self) -> List[str]:
        pass


class StudentAgentInterface(ABC):
    @abstractmethod
    def answer(self, task: Task) -> int:
        """Predict answer to a task (before learning)."""
        pass
    
    @abstractmethod
    def learn(self, task: Task) -> bool:
        """Learn from a task. Returns True if answer was correct."""
        pass
    
    @abstractmethod
    def evaluate(self, eval_tasks: List[Task]) -> float:
        """Evaluate on held-out test set. Returns accuracy (0.0-1.0)."""
        pass
    
    @abstractmethod
    def get_state(self) -> StudentState:
        """Get current state for teacher to observe."""
        pass
    
    @abstractmethod
    def advance_time(self, delta: float = 1.0):
        """Advance time for forgetting simulation."""
        pass


class TeacherAgentInterface(ABC):
    @abstractmethod
    def select_action(self, student_state: StudentState) -> TeacherAction:
        pass
    
    @abstractmethod
    def update(self, action: TeacherAction, reward: float):
        pass
    
    @abstractmethod
    def get_statistics(self) -> Dict:
        pass

