"""Shared data structures and interfaces for Teacher Agent system."""

from dataclasses import dataclass
from typing import List, Dict
from abc import ABC, abstractmethod


@dataclass
class Task:
    """A reading comprehension task."""
    passage: str
    question: str
    choices: List[str]  # 4 choices
    answer: int  # Index 0-3
    topic: str  # e.g., 'history', 'science'
    difficulty: str  # 'easy', 'medium', 'hard'
    task_id: str


@dataclass
class StudentState:
    """Student's current learning state."""
    topic_accuracies: Dict[str, float]  # topic -> accuracy
    topic_attempts: Dict[str, int]
    time_since_practice: Dict[str, float]
    total_timesteps: int
    current_time: float


@dataclass
class TeacherAction:
    """Teacher's decision."""
    topic: str
    difficulty: str
    is_review: bool


class TaskGeneratorInterface(ABC):
    """Interface for task generators."""
    
    @abstractmethod
    def generate_task(self, topic: str, difficulty: str) -> Task:
        """Generate a task for the given topic and difficulty."""
        pass
    
    @abstractmethod
    def get_available_topics(self) -> List[str]:
        """Return list of available topics."""
        pass
    
    @abstractmethod
    def get_available_difficulties(self) -> List[str]:
        """Return list of available difficulties."""
        pass


class StudentAgentInterface(ABC):
    """Interface for student agents."""
    
    @abstractmethod
    def answer(self, task: Task) -> int:
        """Answer a task. Returns index of chosen answer (0-3)."""
        pass
    
    @abstractmethod
    def learn(self, task: Task) -> bool:
        """Learn from a task. Returns whether answer was correct."""
        pass
    
    @abstractmethod
    def evaluate(self, eval_tasks: List[Task]) -> float:
        """Evaluate student on a list of tasks. Returns accuracy (0-1)."""
        pass
    
    @abstractmethod
    def get_state(self) -> StudentState:
        """Get current student state."""
        pass
    
    @abstractmethod
    def advance_time(self, delta: float = 1.0):
        """Advance time for forgetting simulation."""
        pass


class TeacherAgentInterface(ABC):
    """Interface for teacher agents."""
    
    @abstractmethod
    def select_action(self, student_state: StudentState) -> TeacherAction:
        """Select next action based on student state."""
        pass
    
    @abstractmethod
    def update(self, action: TeacherAction, reward: float):
        """Update teacher policy based on reward."""
        pass
    
    @abstractmethod
    def get_statistics(self) -> Dict:
        """Get teacher statistics for visualization."""
        pass

