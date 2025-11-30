"""Simple mock task generator for teacher agent development."""

import random
from typing import List
from interfaces import Task, TaskGeneratorInterface


class MockTaskGenerator(TaskGeneratorInterface):
    """Simple task generator with template-based questions."""
    
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.task_counter = 0
        
        # Template passages and questions
        self.templates = {
            'history': {
                'easy': {
                    'passage': "The Industrial Revolution began in Britain in the late 18th century.",
                    'question': "When did the Industrial Revolution begin?",
                    'correct': "Late 18th century",
                    'distractors': ['Early 19th century', '17th century', 'Middle Ages']
                },
                'medium': {
                    'passage': "The Magna Carta was signed in 1215, establishing principles of limited government.",
                    'question': "What principle did the Magna Carta establish?",
                    'correct': "Limited government",
                    'distractors': ['Absolute monarchy', 'Direct democracy', 'Anarchy']
                },
                'hard': {
                    'passage': "The Peloponnesian War (431-404 BCE) was fought between Athens and Sparta, resulting in Spartan victory and the decline of Athenian democracy.",
                    'question': "What was a major consequence of the Peloponnesian War?",
                    'correct': "Decline of Athenian democracy",
                    'distractors': ['Rise of Roman Empire', 'Formation of Greek city-states', 'Birth of Christianity']
                }
            },
            'science': {
                'easy': {
                    'passage': "Water freezes at 0 degrees Celsius and boils at 100 degrees Celsius.",
                    'question': "At what temperature does water boil?",
                    'correct': "100 degrees Celsius",
                    'distractors': ['0 degrees Celsius', '50 degrees Celsius', '200 degrees Celsius']
                },
                'medium': {
                    'passage': "Photosynthesis is the process by which plants convert light energy into chemical energy stored in glucose.",
                    'question': "What is the main product of photosynthesis?",
                    'correct': "Glucose",
                    'distractors': ['Oxygen', 'Carbon dioxide', 'Water']
                },
                'hard': {
                    'passage': "Quantum entanglement is a phenomenon where particles become correlated such that measuring one instantly affects the other, regardless of distance.",
                    'question': "What property of entangled particles violates classical physics?",
                    'correct': "Instant correlation at distance",
                    'distractors': ['Energy conservation', 'Mass equivalence', 'Wave-particle duality']
                }
            },
            'literature': {
                'easy': {
                    'passage': "Shakespeare wrote Romeo and Juliet, a tragic love story set in Verona.",
                    'question': "Who wrote Romeo and Juliet?",
                    'correct': "Shakespeare",
                    'distractors': ['Dante', 'Homer', 'Virgil']
                },
                'medium': {
                    'passage': "The Great Gatsby by F. Scott Fitzgerald critiques the American Dream through the lens of 1920s excess.",
                    'question': "What theme does The Great Gatsby primarily explore?",
                    'correct': "Critique of the American Dream",
                    'distractors': ['Nature vs nurture', 'Coming of age', 'War and peace']
                },
                'hard': {
                    'passage': "Stream of consciousness narrative technique, used by authors like Virginia Woolf and James Joyce, attempts to capture the continuous flow of thoughts and feelings.",
                    'question': "What does stream of consciousness narrative attempt to capture?",
                    'correct': "Continuous flow of thoughts and feelings",
                    'distractors': ['External events', 'Dialogue only', 'Narrator descriptions']
                }
            },
            'geography': {
                'easy': {
                    'passage': "Mount Everest is the highest mountain on Earth, located in the Himalayas.",
                    'question': "Where is Mount Everest located?",
                    'correct': "Himalayas",
                    'distractors': ['Andes', 'Alps', 'Rockies']
                },
                'medium': {
                    'passage': "The Amazon River flows through South America and drains into the Atlantic Ocean, supporting the largest rainforest in the world.",
                    'question': "Into which ocean does the Amazon River drain?",
                    'correct': "Atlantic Ocean",
                    'distractors': ['Pacific Ocean', 'Indian Ocean', 'Arctic Ocean']
                },
                'hard': {
                    'passage': "The Ring of Fire is a major area in the Pacific Ocean basin where many earthquakes and volcanic eruptions occur due to tectonic plate boundaries.",
                    'question': "Why do earthquakes occur frequently in the Ring of Fire?",
                    'correct': "Tectonic plate boundaries",
                    'distractors': ['Climate change', 'Ocean currents', 'Solar activity']
                }
            },
            'current_events': {
                'easy': {
                    'passage': "Climate change refers to long-term changes in global temperature and weather patterns.",
                    'question': "What does climate change primarily refer to?",
                    'correct': "Long-term changes in global temperature",
                    'distractors': ['Daily weather', 'Seasonal changes', 'Local storms']
                },
                'medium': {
                    'passage': "Renewable energy sources like solar and wind power are becoming more cost-effective alternatives to fossil fuels.",
                    'question': "What makes renewable energy increasingly attractive?",
                    'correct': "Cost-effectiveness",
                    'distractors': ['Availability only', 'Storage capacity', 'Transportation ease']
                },
                'hard': {
                    'passage': "Artificial intelligence ethics concerns questions about bias, privacy, and the societal impact of autonomous decision-making systems.",
                    'question': "What is a primary concern in AI ethics?",
                    'correct': "Bias in decision-making",
                    'distractors': ['Computing power', 'Battery life', 'Network speed']
                }
            }
        }
    
    def get_available_topics(self) -> List[str]:
        """Return list of available topics."""
        return list(self.templates.keys())
    
    def get_available_difficulties(self) -> List[str]:
        """Return list of available difficulties."""
        return ['easy', 'medium', 'hard']
    
    def generate_task(self, topic: str, difficulty: str) -> Task:
        """Generate a task for the given topic and difficulty."""
        if topic not in self.templates:
            raise ValueError(f"Unknown topic: {topic}")
        if difficulty not in self.get_available_difficulties():
            raise ValueError(f"Unknown difficulty: {difficulty}")
        
        template = self.templates[topic][difficulty]
        
        # Create choices: correct answer + distractors
        choices = [template['correct']] + template['distractors']
        self.rng.shuffle(choices)  # Randomize order
        
        # Find correct answer index after shuffling
        correct_idx = choices.index(template['correct'])
        
        self.task_counter += 1
        task_id = f"{topic}_{difficulty}_{self.task_counter}"
        
        return Task(
            passage=template['passage'],
            question=template['question'],
            choices=choices,
            answer=correct_idx,
            topic=topic,
            difficulty=difficulty,
            task_id=task_id
        )

