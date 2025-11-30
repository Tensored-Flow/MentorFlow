"""
Simple mock task generator for independent student testing.
"""

from interfaces import TaskGeneratorInterface, Task
import random


class MockTaskGenerator(TaskGeneratorInterface):
    """Simple task generator with templates."""
    
    def __init__(self):
        self.topics = ['history', 'science', 'literature', 'geography', 'current_events']
        self.difficulties = ['easy', 'medium', 'hard']
        
        self.passages = {
            'history': "The Industrial Revolution began in Britain in the late 18th century. It brought major changes to manufacturing and society.",
            'science': "Photosynthesis is the process by which plants use sunlight to convert carbon dioxide and water into glucose and oxygen.",
            'literature': "Shakespeare wrote numerous plays including Hamlet, Romeo and Juliet, and Macbeth during the Elizabethan era.",
            'geography': "The Amazon rainforest is the world's largest tropical rainforest, spanning nine countries in South America.",
            'current_events': "Artificial intelligence is rapidly advancing, with applications in healthcare, transportation, and education."
        }
        
        self.task_counter = 0
    
    def generate_task(self, topic: str, difficulty: str) -> Task:
        passage = self.passages.get(topic, f"This is a passage about {topic}.")
        
        questions = {
            'easy': f"What is the main topic of this passage?",
            'medium': f"What can be inferred from this passage about {topic}?",
            'hard': f"Which statement best synthesizes the information in this passage?"
        }
        
        question = questions[difficulty]
        
        # Generate choices
        correct = f"It discusses {topic}"
        wrong = [
            f"It's primarily about a different subject",
            f"The passage focuses on unrelated matters",
            f"This is not the main theme"
        ]
        
        choices = [correct] + wrong
        answer_idx = 0
        
        # Shuffle
        combined = list(enumerate(choices))
        random.shuffle(combined)
        answer_idx = [i for i, (orig, _) in enumerate(combined) if orig == 0][0]
        choices = [c for _, c in combined]
        
        self.task_counter += 1
        
        return Task(
            passage=passage,
            question=question,
            choices=choices,
            answer=answer_idx,
            topic=topic,
            difficulty=difficulty,
            task_id=f"{topic}_{difficulty}_{self.task_counter}"
        )
    
    def get_available_topics(self):
        return self.topics
    
    def get_available_difficulties(self):
        return self.difficulties

