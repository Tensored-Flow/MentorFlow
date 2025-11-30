"""
DistilBERT-based student agent with online learning and memory decay.

Uses DistilBERT for Multiple Choice to answer reading comprehension tasks.
Implements online learning (fine-tune on 1 example at a time).
"""

import torch
from torch.optim import AdamW
from transformers import (
    DistilBertForMultipleChoice,
    DistilBertTokenizer,
)
from typing import List, Dict
import numpy as np
from collections import defaultdict

from interfaces import StudentAgentInterface, StudentState, Task
from memory_decay import MemoryDecayModel


class StudentAgent(StudentAgentInterface):
    """
    DistilBERT-based student that learns reading comprehension.
    
    Features:
    - Online learning (1 example at a time)
    - Memory decay (Ebbinghaus forgetting)
    - Per-topic skill tracking
    - Gradient accumulation for stability
    """
    
    def __init__(
        self,
        learning_rate: float = 5e-5,
        retention_constant: float = 80.0,
        device: str = 'cpu',
        max_length: int = 256,
        gradient_accumulation_steps: int = 4
    ):
        """
        Args:
            learning_rate: LM fine-tuning learning rate
            retention_constant: Forgetting speed (higher = slower forgetting)
            device: 'cpu' or 'cuda'
            max_length: Max tokens for passage + question + choices
            gradient_accumulation_steps: Accumulate gradients for stability
        """
        self.device = device
        self.max_length = max_length
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        # Load DistilBERT for multiple choice
        # Allow silent mode for testing
        verbose = True  # Can be overridden
        
        try:
            if verbose:
                print("Loading DistilBERT model...", end=" ", flush=True)
            self.model = DistilBertForMultipleChoice.from_pretrained(
                "distilbert-base-uncased"
            ).to(self.device)
            
            self.tokenizer = DistilBertTokenizer.from_pretrained(
                "distilbert-base-uncased"
            )
            if verbose:
                print("✅")
        except Exception as e:
            if verbose:
                print(f"⚠️ (Model unavailable, using dummy mode)")
            self.model = None
            self.tokenizer = None
        
        # Optimizer
        if self.model:
            self.optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        else:
            self.optimizer = None
        
        # Memory decay model
        self.memory = MemoryDecayModel(retention_constant=retention_constant)
        
        # Track per-topic base skills (before forgetting)
        self.topic_base_skills: Dict[str, float] = {}
        
        # Track learning history
        self.topic_attempts: Dict[str, int] = defaultdict(int)
        self.topic_correct: Dict[str, int] = defaultdict(int)
        
        # Gradient accumulation counter
        self.grad_step = 0
        
        # Training mode flag
        if self.model:
            self.model.train()
    
    def answer(self, task: Task) -> int:
        """
        Predict answer without updating weights.
        
        Prediction accuracy is modulated by memory decay.
        """
        if not self.model:
            # Dummy model: random guessing
            return np.random.randint(0, 4)
        
        self.model.eval()
        
        # Prepare inputs
        inputs = self._prepare_inputs(task)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            predicted_idx = torch.argmax(logits, dim=-1).item()
        
        # Apply memory decay to prediction
        # If student has forgotten, prediction becomes more random
        effective_skill = self.memory.get_effective_skill(task.topic)
        
        # Probability of using learned answer vs random guess
        # MCQ baseline = 0.25 (random guessing)
        use_learned_prob = 0.25 + 0.75 * effective_skill
        
        if np.random.random() < use_learned_prob:
            return predicted_idx
        else:
            # Random guess
            return np.random.randint(0, 4)
    
    def learn(self, task: Task) -> bool:
        """
        Fine-tune on a single task (online learning).
        
        Returns:
            True if prediction was correct, False otherwise
        """
        if not self.model:
            # Dummy learning: track statistics only
            predicted = np.random.randint(0, 4)
            was_correct = (predicted == task.answer)
            self._update_stats(task, was_correct)
            return was_correct
        
        self.model.train()
        
        # Get prediction before learning
        predicted = self.answer(task)
        was_correct = (predicted == task.answer)
        
        # Prepare inputs with correct answer
        inputs = self._prepare_inputs(task)
        inputs['labels'] = torch.tensor([task.answer], device=self.device)
        
        # Forward pass
        outputs = self.model(**inputs)
        loss = outputs.loss
        
        # Backward pass with gradient accumulation
        loss = loss / self.gradient_accumulation_steps
        loss.backward()
        
        self.grad_step += 1
        
        # Update weights every N steps
        if self.grad_step % self.gradient_accumulation_steps == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()
        
        # Update statistics
        self._update_stats(task, was_correct)
        
        return was_correct
    
    def _update_stats(self, task: Task, was_correct: bool):
        """Update topic statistics and memory."""
        self.topic_attempts[task.topic] += 1
        if was_correct:
            self.topic_correct[task.topic] += 1
        
        # Compute base skill (accuracy without forgetting)
        base_skill = self.topic_correct[task.topic] / self.topic_attempts[task.topic]
        self.topic_base_skills[task.topic] = base_skill
        
        # Update memory (record practice)
        self.memory.update_practice(task.topic, base_skill)
    
    def evaluate(self, eval_tasks: List[Task]) -> float:
        """
        Evaluate on held-out tasks without updating weights.
        
        Returns:
            Accuracy (0.0-1.0)
        """
        if not eval_tasks:
            return 0.0
        
        if not self.model:
            # Dummy evaluation: return random
            return 0.25
        
        self.model.eval()
        
        correct = 0
        for task in eval_tasks:
            predicted = self.answer(task)
            if predicted == task.answer:
                correct += 1
        
        return correct / len(eval_tasks)
    
    def get_state(self) -> StudentState:
        """
        Get current state for teacher observation.
        
        Returns per-topic accuracies accounting for forgetting.
        """
        topic_accuracies = {}
        time_since_practice = {}
        
        for topic in self.topic_base_skills:
            # Get effective skill (with forgetting)
            effective_skill = self.memory.get_effective_skill(topic)
            
            # Convert to expected accuracy on MCQ
            topic_accuracies[topic] = 0.25 + 0.75 * effective_skill
            
            # Time since last practice
            time_since_practice[topic] = self.memory.get_time_since_practice(topic)
        
        return StudentState(
            topic_accuracies=topic_accuracies,
            topic_attempts=dict(self.topic_attempts),
            time_since_practice=time_since_practice,
            total_timesteps=sum(self.topic_attempts.values()),
            current_time=self.memory.current_time
        )
    
    def _prepare_inputs(self, task: Task) -> Dict[str, torch.Tensor]:
        """
        Prepare inputs for DistilBERT multiple choice model.
        
        Format: [CLS] passage [SEP] question [SEP] choice [SEP]
        Repeated for each of 4 choices.
        """
        if not self.tokenizer:
            return {}
        
        # Create 4 input sequences (one per choice)
        input_texts = []
        for choice in task.choices:
            # Format: passage + question + choice
            text = f"{task.passage} {task.question} {choice}"
            input_texts.append(text)
        
        # Tokenize
        encoded = self.tokenizer(
            input_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Reshape for multiple choice format
        # (batch_size=1, num_choices=4, seq_length)
        input_ids = encoded['input_ids'].unsqueeze(0).to(self.device)
        attention_mask = encoded['attention_mask'].unsqueeze(0).to(self.device)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
    
    def advance_time(self, delta: float = 1.0):
        """Advance time for memory decay."""
        self.memory.advance_time(delta)
    
    def save(self, path: str):
        """Save model checkpoint."""
        if not self.model:
            print("⚠️ No model to save (using dummy model)")
            return
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'topic_base_skills': self.topic_base_skills,
            'topic_attempts': dict(self.topic_attempts),
            'topic_correct': dict(self.topic_correct),
            'memory': self.memory,
            'grad_step': self.grad_step
        }, path)
        print(f"💾 Saved checkpoint to {path}")
    
    def load(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        if self.model:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            if self.optimizer and checkpoint.get('optimizer_state_dict'):
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        self.topic_base_skills = checkpoint['topic_base_skills']
        self.topic_attempts = defaultdict(int, checkpoint['topic_attempts'])
        self.topic_correct = defaultdict(int, checkpoint['topic_correct'])
        self.memory = checkpoint['memory']
        self.grad_step = checkpoint.get('grad_step', 0)
        print(f"✅ Loaded checkpoint from {path}")

