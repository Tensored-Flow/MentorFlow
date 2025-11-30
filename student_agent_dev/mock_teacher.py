"""
Simple mock teacher agent for testing student independently.
"""

from interfaces import TeacherAgentInterface, TeacherAction, StudentState
import random


class MockTeacherAgent(TeacherAgentInterface):
    """Simple random teacher for testing student independently."""
    
    def __init__(self):
        self.topics = ['history', 'science', 'literature', 'geography', 'current_events']
        self.difficulties = ['easy', 'medium', 'hard']
    
    def select_action(self, student_state: StudentState) -> TeacherAction:
        # Strategy: slightly intelligent curriculum
        # Start with easy, gradually increase difficulty
        
        if student_state.total_timesteps < 20:
            difficulty = 'easy'
        elif student_state.total_timesteps < 100:
            difficulty = random.choice(['easy', 'medium'])
        else:
            difficulty = random.choice(['medium', 'hard'])
        
        topic = random.choice(self.topics)
        is_review = random.random() < 0.2  # 20% chance of review
        
        return TeacherAction(topic=topic, difficulty=difficulty, is_review=is_review)
    
    def update(self, action: TeacherAction, reward: float):
        pass  # Mock doesn't learn
    
    def get_statistics(self) -> dict:
        return {}

