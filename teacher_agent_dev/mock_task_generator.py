"""Expanded mock task generator with many families and multiple difficulty levels."""

import random
from typing import List, Tuple
from interfaces import Task, TaskGeneratorInterface


class MockTaskGenerator(TaskGeneratorInterface):
    """
    Expanded task generator with:
    - 15+ topic families
    - 5-7 difficulty levels (higher = multi-step)
    - Procedural task generation
    """
    
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.task_counter = 0
        
        # Expanded topic families (15+ topics)
        self.topics = [
            'history', 'science', 'literature', 'geography', 'current_events',
            'mathematics', 'programming', 'philosophy', 'art', 'music',
            'biology', 'chemistry', 'physics', 'economics', 'psychology'
        ]
        
        # Expanded difficulty levels (5-7 levels)
        # Higher levels involve multi-step reasoning
        self.difficulties = [
            'trivial',      # 0: Single fact recall
            'easy',         # 1: Simple understanding
            'medium',       # 2: Application of concepts
            'hard',         # 3: Analysis and reasoning (2-3 steps)
            'expert',       # 4: Complex multi-step (3-4 steps)
            'master',       # 5: Advanced multi-step (4-5 steps)
            'grandmaster'   # 6: Expert-level synthesis (5+ steps)
        ]
        
        # Template structure for each topic
        self._init_templates()
    
    def _init_templates(self):
        """Initialize template structures for procedural generation."""
        # Templates store base patterns, not fixed questions
        self.template_patterns = {
            topic: {
                'base_concepts': self._get_base_concepts(topic),
                'relationships': self._get_relationships(topic),
                'complexity_factors': self._get_complexity_factors(topic)
            }
            for topic in self.topics
        }
    
    def _get_base_concepts(self, topic: str) -> List[str]:
        """Get base concepts for a topic."""
        concept_map = {
            'history': ['dates', 'events', 'causes', 'effects', 'figures'],
            'science': ['principles', 'laws', 'experiments', 'observations'],
            'literature': ['themes', 'symbols', 'characters', 'plot', 'style'],
            'geography': ['locations', 'features', 'climate', 'resources'],
            'current_events': ['trends', 'issues', 'policies', 'impacts'],
            'mathematics': ['operations', 'equations', 'patterns', 'proofs'],
            'programming': ['syntax', 'algorithms', 'data structures', 'patterns'],
            'philosophy': ['concepts', 'arguments', 'theories', 'ethics'],
            'art': ['styles', 'techniques', 'movements', 'artists'],
            'music': ['theory', 'instruments', 'genres', 'composers'],
            'biology': ['cells', 'systems', 'processes', 'evolution'],
            'chemistry': ['elements', 'reactions', 'bonding', 'mechanisms'],
            'physics': ['forces', 'energy', 'fields', 'particles'],
            'economics': ['markets', 'policies', 'indicators', 'theories'],
            'psychology': ['behavior', 'cognition', 'theories', 'methods']
        }
        return concept_map.get(topic, ['concept1', 'concept2', 'concept3'])
    
    def _get_relationships(self, topic: str) -> List[str]:
        """Get relationship types for multi-step reasoning."""
        return ['causes', 'enables', 'requires', 'leads_to', 'depends_on', 'influences']
    
    def _get_complexity_factors(self, topic: str) -> List[str]:
        """Get factors that increase complexity."""
        return ['context', 'exceptions', 'interactions', 'historical', 'contemporary']
    
    def get_available_topics(self) -> List[str]:
        """Return list of available topics."""
        return self.topics.copy()
    
    def get_available_difficulties(self) -> List[str]:
        """Return list of available difficulties."""
        return self.difficulties.copy()
    
    def _get_steps_for_difficulty(self, difficulty: str) -> int:
        """Determine number of reasoning steps for a difficulty level."""
        step_map = {
            'trivial': 1,
            'easy': 1,
            'medium': 2,
            'hard': 3,
            'expert': 4,
            'master': 5,
            'grandmaster': 6
        }
        return step_map.get(difficulty, 1)
    
    def _generate_multi_step_question(self, topic: str, difficulty: str) -> Tuple[str, str, List[str]]:
        """
        Generate a question with multiple reasoning steps.
        
        Returns:
            (passage, question, [correct, distractor1, distractor2, distractor3])
        """
        steps = self._get_steps_for_difficulty(difficulty)
        concepts = self.template_patterns[topic]['base_concepts']
        relationships = self.template_patterns[topic]['relationships']
        
        # Select concepts and relationships based on difficulty
        selected_concepts = self.rng.sample(concepts, min(steps, len(concepts)))
        selected_relationships = self.rng.sample(relationships, steps - 1) if steps > 1 else []
        
        # Generate passage with multi-step reasoning
        passage_parts = []
        question_context = []
        
        for i, concept in enumerate(selected_concepts):
            if i == 0:
                passage_parts.append(f"In {topic}, {concept} is fundamental.")
                question_context.append(concept)
            else:
                rel = selected_relationships[i-1] if i-1 < len(selected_relationships) else 'relates to'
                passage_parts.append(f"{concept} {rel} {selected_concepts[i-1]}.")
                question_context.append(f"{rel} {concept}")
        
        passage = " ".join(passage_parts)
        
        # Generate question that requires multi-step reasoning
        if steps == 1:
            question = f"What is the primary {selected_concepts[0]} in {topic}?"
            correct = f"Primary {selected_concepts[0]}"
        elif steps == 2:
            question = f"Given that {selected_concepts[0]} {selected_relationships[0]} {selected_concepts[1]}, what is the result?"
            correct = f"{selected_concepts[0]} → {selected_concepts[1]}"
        elif steps == 3:
            question = f"If {selected_concepts[0]} leads to {selected_concepts[1]}, and {selected_concepts[1]} influences {selected_concepts[2] if len(selected_concepts) > 2 else selected_concepts[0]}, what is the final outcome?"
            correct = f"Chain: {selected_concepts[0]} → {selected_concepts[1]} → {selected_concepts[min(2, len(selected_concepts)-1)]}"
        else:
            # Complex multi-step
            question = f"Considering the relationship chain: {' → '.join(selected_concepts[:steps])}, what synthesis emerges?"
            correct = f"Synthesis from {steps} steps"
        
        # Generate distractors
        distractors = [
            f"Alternative {selected_concepts[0] if selected_concepts else 'answer'}",
            f"Unrelated concept",
            f"Reverse relationship"
        ]
        
        return passage, question, [correct] + distractors
    
    def generate_task(self, topic: str, difficulty: str) -> Task:
        """Generate a task for the given topic and difficulty."""
        if topic not in self.topics:
            raise ValueError(f"Unknown topic: {topic}. Available: {self.topics}")
        if difficulty not in self.difficulties:
            raise ValueError(f"Unknown difficulty: {difficulty}. Available: {self.difficulties}")
        
        # Try topic-specific generator first, fall back to generic
        templates = {
            'history': self._generate_history_question,
            'science': self._generate_science_question,
            'mathematics': self._generate_math_question,
            'programming': self._generate_programming_question,
        }
        
        generator = templates.get(topic)
        if generator:
            passage, question, choices_list = generator(difficulty)
        else:
            passage, question, choices_list = self._generate_multi_step_question(topic, difficulty)
        
        # Shuffle choices
        correct_answer = choices_list[0]  # First is always correct
        self.rng.shuffle(choices_list)
        correct_idx = choices_list.index(correct_answer)
        
        # Create task ID
        self.task_counter += 1
        task_id = f"{topic}_{difficulty}_{self.task_counter}"
        
        return Task(
            passage=passage,
            question=question,
            choices=choices_list,
            answer=correct_idx,
            topic=topic,
            difficulty=difficulty,
            task_id=task_id
        )
    
    def _generate_topic_specific_question(self, topic: str, difficulty: str) -> Tuple[str, str, List[str]]:
        """Generate topic-specific question templates for more realistic tasks."""
        templates = {
            'history': self._generate_history_question,
            'science': self._generate_science_question,
            'mathematics': self._generate_math_question,
            'programming': self._generate_programming_question,
        }
        
        generator = templates.get(topic, self._generate_generic_question)
        return generator(difficulty)
    
    def _generate_history_question(self, difficulty: str) -> Tuple[str, str, List[str]]:
        """Generate history-specific questions."""
        events = [
            ("Industrial Revolution", "Britain", "late 18th century"),
            ("World War II", "1939-1945", "global conflict"),
            ("Renaissance", "Italy", "14th-17th century"),
            ("French Revolution", "1789", "socio-political upheaval"),
            ("Cold War", "1947-1991", "ideological conflict")
        ]
        
        event = self.rng.choice(events)
        steps = self._get_steps_for_difficulty(difficulty)
        
        if steps == 1:
            passage = f"The {event[0]} began in {event[1]}."
            question = f"When did the {event[0]} occur?"
            correct = event[1] if 'century' in event[1] or len(event[1]) > 4 else event[2]
        elif steps == 2:
            passage = f"The {event[0]} started in {event[1]} and led to {event[2]}."
            question = f"What was a major consequence of the {event[0]}?"
            correct = event[2]
        else:
            passage = f"The {event[0]} began in {event[1]}, caused {event[2]}, and influenced subsequent historical developments."
            question = f"What sequence of effects did the {event[0]} create?"
            correct = f"{event[1]} → {event[2]} → Historical changes"
        
        distractors = [
            f"Alternative historical period",
            f"Different region",
            f"Unrelated event"
        ]
        
        return passage, question, [correct] + distractors
    
    def _generate_science_question(self, difficulty: str) -> Tuple[str, str, List[str]]:
        """Generate science-specific questions."""
        concepts = [
            ("Photosynthesis", "converts light to glucose", "requires CO2 and H2O"),
            ("Evolution", "natural selection", "genetic variation"),
            ("Gravity", "attracts mass", "affects motion")
        ]
        
        concept = self.rng.choice(concepts)
        steps = self._get_steps_for_difficulty(difficulty)
        
        if steps == 1:
            passage = f"{concept[0]} is a fundamental process."
            question = f"What does {concept[0]} do?"
            correct = concept[1]
        elif steps == 2:
            passage = f"{concept[0]} {concept[1]} and {concept[2]}."
            question = f"How does {concept[0]} work?"
            correct = f"{concept[1]} using {concept[2]}"
        else:
            passage = f"{concept[0]} {concept[1]}. This process {concept[2]}, which enables further biological processes."
            question = f"What is the complete mechanism of {concept[0]}?"
            correct = f"{concept[1]} → {concept[2]} → Biological outcomes"
        
        distractors = [
            "Different mechanism",
            "Incorrect process",
            "Unrelated concept"
        ]
        
        return passage, question, [correct] + distractors
    
    def _generate_math_question(self, difficulty: str) -> Tuple[str, str, List[str]]:
        """Generate mathematics questions with varying complexity."""
        steps = self._get_steps_for_difficulty(difficulty)
        
        if steps == 1:
            a, b = self.rng.randint(1, 10), self.rng.randint(1, 10)
            passage = f"Consider the numbers {a} and {b}."
            question = f"What is {a} + {b}?"
            correct = str(a + b)
        elif steps == 2:
            a, b, c = self.rng.randint(1, 10), self.rng.randint(1, 10), self.rng.randint(1, 10)
            passage = f"Given: x = {a}, y = {b}, z = {c}."
            question = f"What is (x + y) * z?"
            correct = str((a + b) * c)
        elif steps == 3:
            a, b, c, d = [self.rng.randint(1, 5) for _ in range(4)]
            passage = f"Given: a={a}, b={b}, c={c}, d={d}. Compute: a*b, then add c, then multiply by d."
            question = f"What is the final result?"
            correct = str((a * b + c) * d)
        else:
            # Multi-step algebraic chain
            values = [self.rng.randint(1, 5) for _ in range(steps + 1)]
            passage = f"Given values: {', '.join([f'v{i}={values[i]}' for i in range(len(values))])}"
            question = f"Compute: v0 * v1 + v2 * v3 - v4 (if applicable)"
            result = values[0] * values[1] + (values[2] * values[3] if len(values) > 3 else 0) - (values[4] if len(values) > 4 else 0)
            correct = str(result)
        
        distractors = [
            str(self.rng.randint(0, 100)),
            str(self.rng.randint(0, 100)),
            str(self.rng.randint(0, 100))
        ]
        
        return passage, question, [correct] + distractors
    
    def _generate_programming_question(self, difficulty: str) -> Tuple[str, str, List[str]]:
        """Generate programming questions."""
        steps = self._get_steps_for_difficulty(difficulty)
        
        if steps == 1:
            passage = "In Python, list indexing starts at 0."
            question = "What is the first index of a list?"
            correct = "0"
        elif steps == 2:
            passage = "Consider: arr = [1, 2, 3, 4, 5]. First, get arr[1:3], then access the last element."
            question = "What is the result?"
            correct = "3"
        elif steps == 3:
            passage = "Code: x = [1, 2, 3]; y = x[1:]; z = y[-1] + x[0]"
            question = "What is z?"
            correct = "4"  # y[-1] = 3, x[0] = 1, so 3+1=4
        else:
            # Multi-step: a = [1,2,3,4]; b = a[1:3]; c = sum(b); d = c * a[0]
            # a[1:3] = [2,3], sum(b) = 5, a[0] = 1, so d = 5 * 1 = 5
            passage = "Multi-step: a = [1,2,3,4]; b = a[1:3]; c = sum(b); d = c * a[0]"
            question = "What is d?"
            correct = "5"  # a[1:3]=[2,3], sum=5, 5*1=5
        
        distractors = ["0", "1", "2"]
        
        return passage, question, [correct] + distractors
    
    def _generate_generic_question(self, difficulty: str) -> Tuple[str, str, List[str]]:
        """Fallback generic question generator."""
        return self._generate_multi_step_question(self.rng.choice(self.topics), difficulty)
