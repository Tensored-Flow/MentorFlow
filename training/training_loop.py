"""
Meta-Training Loop for TeachRL
Integrates Teacher (bandit) and Student (PPO) in the curriculum learning loop.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
import json
import time

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    tqdm = None  # Will be checked before use

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from tasks.task_generator import (
    generate_eval_dataset, generate_task_by_arm, arm_to_name,
    NUM_FAMILIES, NUM_DIFFICULTIES, TaskSpec
)
from student.ppo_agent import StudentAgent
from teacher.teacher_bandit import (
    TeacherBandit, BanditStrategy, RandomTeacher, FixedCurriculumTeacher
)


@dataclass
class MetaTrainingConfig:
    """Configuration for meta-training."""
    # Meta-loop settings
    num_meta_steps: int = 200
    student_train_steps: int = 256  # PPO steps per meta-step
    eval_tasks_per_type: int = 10   # Tasks per type in eval set
    
    # Teacher settings
    teacher_strategy: BanditStrategy = BanditStrategy.UCB1
    exploration_param: float = 2.0
    epsilon: float = 0.1
    
    # Student settings
    learning_rate: float = 3e-4
    ppo_n_steps: int = 128
    ppo_batch_size: int = 64
    
    # Reproducibility
    seed: int = 42
    
    # Logging
    log_every: int = 10
    verbose: bool = True


@dataclass
class MetaTrainingResults:
    """Results from meta-training run."""
    # Per-step data
    meta_steps: List[int] = field(default_factory=list)
    student_accuracies: List[float] = field(default_factory=list)
    teacher_rewards: List[float] = field(default_factory=list)
    selected_arms: List[int] = field(default_factory=list)
    
    # Final statistics
    final_accuracy: float = 0.0
    total_teacher_reward: float = 0.0
    curriculum_heatmap: Optional[np.ndarray] = None
    accuracy_by_type: Optional[Dict[str, float]] = None
    
    # Timing
    total_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "meta_steps": self.meta_steps,
            "student_accuracies": self.student_accuracies,
            "teacher_rewards": self.teacher_rewards,
            "selected_arms": self.selected_arms,
            "final_accuracy": self.final_accuracy,
            "total_teacher_reward": self.total_teacher_reward,
            "curriculum_heatmap": self.curriculum_heatmap.tolist() if self.curriculum_heatmap is not None else None,
            "accuracy_by_type": self.accuracy_by_type,
            "total_time": self.total_time,
        }
    
    def save(self, path: str):
        """Save results to JSON."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


def run_random_curriculum(
    meta_steps: int,
    model: StudentAgent,
    env: Any,
    eval_set: List[TaskSpec],
    num_arms: int,
    rng: Any
) -> Dict[str, Any]:
    """Run baseline where tasks are selected uniformly at random."""
    heatmap = [[0 for _ in range(meta_steps)] for _ in range(num_arms)]
    accuracies: List[float] = []
    selected_arms: List[int] = []
    train_steps = getattr(env, "student_train_steps", None)
    if train_steps is None and isinstance(env, dict):
        train_steps = env.get("train_steps")
    if train_steps is None:
        train_steps = getattr(env, "train_steps", 256)
    
    for step in range(meta_steps):
        arm_id = rng.randint(0, num_arms - 1)
        selected_arms.append(arm_id)
        heatmap[arm_id][step] = 1
        model.train_on_task(arm_id=arm_id, total_timesteps=train_steps)
        accuracy = model.evaluate(eval_set)
        accuracies.append(accuracy)
    
    return {
        "accuracy": accuracies,
        "heatmap": heatmap,
        "arms": selected_arms,
        "meta_steps": meta_steps,
    }


def run_fixed_curriculum(
    meta_steps: int,
    model: StudentAgent,
    env: Any,
    eval_set: List[TaskSpec],
    num_arms: int,
    rng: Any = None
) -> Dict[str, Any]:
    """Run baseline with deterministic easy→medium→hard curriculum."""
    heatmap = [[0 for _ in range(meta_steps)] for _ in range(num_arms)]
    accuracies: List[float] = []
    selected_arms: List[int] = []
    train_steps = getattr(env, "student_train_steps", None)
    if train_steps is None and isinstance(env, dict):
        train_steps = env.get("train_steps")
    if train_steps is None:
        train_steps = getattr(env, "train_steps", 256)

    # Build fixed order: all easy, then medium, then hard
    fixed_order: List[int] = []
    for difficulty in range(NUM_DIFFICULTIES):
        for family in range(NUM_FAMILIES):
            arm_id = family * NUM_DIFFICULTIES + difficulty
            fixed_order.append(arm_id)

    for step in range(meta_steps):
        arm_id = fixed_order[step % len(fixed_order)]
        selected_arms.append(arm_id)
        heatmap[arm_id][step] = 1
        model.train_on_task(arm_id=arm_id, total_timesteps=train_steps)
        accuracy = model.evaluate(eval_set)
        accuracies.append(accuracy)

    return {
        "accuracy": accuracies,
        "heatmap": heatmap,
        "arms": selected_arms,
        "meta_steps": meta_steps,
    }


class MetaTrainer:
    """
    Meta-training loop that coordinates teacher and student.
    
    Loop:
    1. Evaluate student on fixed eval set → acc_before
    2. Teacher selects arm (task type)
    3. Student trains on that task type for K steps
    4. Evaluate student again → acc_after
    5. Teacher receives reward = acc_after - acc_before
    6. Repeat
    """
    
    def __init__(self, config: MetaTrainingConfig):
        self.config = config
        
        # Create evaluation dataset (fixed for consistency)
        self.eval_tasks = generate_eval_dataset(
            tasks_per_type=config.eval_tasks_per_type,
            seed=config.seed
        )
        
        # Initialize student
        self.student = StudentAgent(
            learning_rate=config.learning_rate,
            n_steps=config.ppo_n_steps,
            batch_size=config.ppo_batch_size,
            seed=config.seed,
            verbose=0
        )
        
        # Initialize teacher
        self.teacher = self._create_teacher(config.teacher_strategy)
        
        # Results tracking
        self.results = MetaTrainingResults()
    
    def _create_teacher(self, strategy: Union[BanditStrategy, str]):
        """Create teacher based on strategy."""
        if isinstance(strategy, str):
            if strategy == "random":
                return RandomTeacher(seed=self.config.seed)
            elif strategy == "fixed":
                return FixedCurriculumTeacher(
                    steps_per_difficulty=self.config.num_meta_steps // 3,
                    seed=self.config.seed
                )
            strategy = BanditStrategy(strategy)
        
        if strategy == BanditStrategy.RANDOM:
            return RandomTeacher(seed=self.config.seed)
        
        return TeacherBandit(
            strategy=strategy,
            exploration_param=self.config.exploration_param,
            epsilon=self.config.epsilon,
            seed=self.config.seed
        )
    
    def train(self) -> MetaTrainingResults:
        """
        Run the meta-training loop.
        
        Returns:
            MetaTrainingResults with all logged data
        """
        start_time = time.time()
        
        if self.config.verbose:
            print("=" * 60)
            print("TeachRL Meta-Training")
            print("=" * 60)
            print(f"Meta-steps: {self.config.num_meta_steps}")
            print(f"Student train steps per meta-step: {self.config.student_train_steps}")
            print(f"Eval dataset size: {len(self.eval_tasks)}")
            print(f"Teacher strategy: {self.config.teacher_strategy}")
            print("=" * 60)
        
        # Progress bar for meta-training iterations
        iterator = range(self.config.num_meta_steps)
        if HAS_TQDM and self.config.verbose:
            iterator = tqdm(iterator, desc="Meta-training", unit="step")
        
        for step in iterator:
            # 1. Evaluate student before training
            acc_before = self.student.evaluate(self.eval_tasks)
            
            # 2. Teacher selects task type
            arm_id = self.teacher.select_arm()
            
            # 3. Student trains on selected task
            self.student.train_on_task(
                arm_id=arm_id,
                total_timesteps=self.config.student_train_steps
            )
            
            # 4. Evaluate student after training
            acc_after = self.student.evaluate(self.eval_tasks)
            
            # 5. Compute teacher reward (improvement)
            reward = acc_after - acc_before
            self.teacher.update(arm_id, reward)
            
            # 6. Log results
            self.results.meta_steps.append(step)
            self.results.student_accuracies.append(acc_after)
            self.results.teacher_rewards.append(reward)
            self.results.selected_arms.append(arm_id)
            
            # Progress logging
            if self.config.verbose and (step + 1) % self.config.log_every == 0:
                avg_reward = np.mean(self.results.teacher_rewards[-self.config.log_every:])
                print(f"Step {step + 1:4d} | "
                      f"Acc: {acc_after:.2%} | "
                      f"Δ: {reward:+.3f} | "
                      f"Arm: {arm_to_name(arm_id):20s} | "
                      f"Avg Reward: {avg_reward:+.4f}")
        
        # Final evaluation and statistics
        self.results.final_accuracy = self.student.evaluate(self.eval_tasks)
        self.results.total_teacher_reward = sum(self.results.teacher_rewards)
        self.results.accuracy_by_type = self.student.evaluate_by_type(self.eval_tasks)
        self.results.total_time = time.time() - start_time
        
        # Get curriculum heatmap if teacher supports it
        if hasattr(self.teacher, 'get_curriculum_heatmap'):
            self.results.curriculum_heatmap = self.teacher.get_curriculum_heatmap()
        else:
            # Build from selection history
            heatmap = np.zeros((NUM_FAMILIES, NUM_DIFFICULTIES))
            for arm in self.results.selected_arms:
                family = arm // NUM_DIFFICULTIES
                difficulty = arm % NUM_DIFFICULTIES
                heatmap[family, difficulty] += 1
            self.results.curriculum_heatmap = heatmap
        
        if self.config.verbose:
            print("=" * 60)
            print("Training Complete!")
            print(f"Final Accuracy: {self.results.final_accuracy:.2%}")
            print(f"Total Teacher Reward: {self.results.total_teacher_reward:.4f}")
            print(f"Time: {self.results.total_time:.1f}s")
            print("=" * 60)
        
        return self.results
    
    def reset(self):
        """Reset trainer for a new run."""
        self.student.reset()
        self.teacher.reset()
        self.results = MetaTrainingResults()


def run_comparison(
    config: MetaTrainingConfig,
    strategies: List[str] = ["ucb1", "thompson_sampling", "random", "fixed"]
) -> Dict[str, MetaTrainingResults]:
    """
    Run meta-training with multiple teacher strategies for comparison.
    
    Args:
        config: Base configuration
        strategies: List of strategies to compare
    
    Returns:
        Dict mapping strategy name to results
    """
    all_results = {}
    
    for strategy in strategies:
        print(f"\n{'='*60}")
        print(f"Running: {strategy.upper()}")
        print(f"{'='*60}")
        
        # Create config for this strategy
        strategy_config = MetaTrainingConfig(
            num_meta_steps=config.num_meta_steps,
            student_train_steps=config.student_train_steps,
            eval_tasks_per_type=config.eval_tasks_per_type,
            teacher_strategy=strategy if strategy in ["random", "fixed"] else BanditStrategy(strategy),
            exploration_param=config.exploration_param,
            epsilon=config.epsilon,
            learning_rate=config.learning_rate,
            ppo_n_steps=config.ppo_n_steps,
            ppo_batch_size=config.ppo_batch_size,
            seed=config.seed,
            log_every=config.log_every,
            verbose=config.verbose
        )
        
        trainer = MetaTrainer(strategy_config)
        results = trainer.train()
        all_results[strategy] = results
    
    # Print comparison summary
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"{'Strategy':<20} {'Final Acc':>12} {'Total Reward':>14} {'Time':>10}")
    print("-" * 60)
    for strategy, results in all_results.items():
        print(f"{strategy:<20} {results.final_accuracy:>11.2%} "
              f"{results.total_teacher_reward:>14.4f} "
              f"{results.total_time:>9.1f}s")
    
    return all_results


# =============================================================================
# STANDALONE TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("TeachRL Meta-Training Loop Test")
    print("=" * 60)
    
    # Quick test config
    config = MetaTrainingConfig(
        num_meta_steps=30,
        student_train_steps=128,
        eval_tasks_per_type=5,
        teacher_strategy=BanditStrategy.UCB1,
        seed=42,
        log_every=5,
        verbose=True
    )
    
    # Run single training
    trainer = MetaTrainer(config)
    results = trainer.train()
    
    # Show accuracy by type
    print("\nAccuracy by task type:")
    for name, acc in sorted(results.accuracy_by_type.items()):
        print(f"  {name}: {acc:.0%}")
    
    # Show curriculum heatmap
    print("\nCurriculum heatmap (task selections):")
    print(results.curriculum_heatmap)
    
    print("\n✓ Meta-training loop test complete!")
