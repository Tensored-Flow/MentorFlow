"""
Experiment Runner for TeachRL
Run full experiments comparing different curriculum strategies.
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.training_loop import (
    MetaTrainer, MetaTrainingConfig, MetaTrainingResults,
    run_comparison
)
from teacher.teacher_bandit import BanditStrategy


def run_full_experiment(
    output_dir: str = "results",
    num_meta_steps: int = 200,
    student_train_steps: int = 256,
    eval_tasks_per_type: int = 10,
    seed: int = 42,
    strategies: Optional[List[str]] = None
) -> Dict[str, MetaTrainingResults]:
    """
    Run a full comparison experiment.
    
    Args:
        output_dir: Directory to save results
        num_meta_steps: Number of meta-training steps
        student_train_steps: PPO steps per meta-step
        eval_tasks_per_type: Tasks per type in eval set
        seed: Random seed
        strategies: List of strategies to compare
    
    Returns:
        Dict mapping strategy to results
    """
    # Default strategies
    if strategies is None:
        strategies = ["ucb1", "thompson_sampling", "epsilon_greedy", "random", "fixed"]
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Base config
    config = MetaTrainingConfig(
        num_meta_steps=num_meta_steps,
        student_train_steps=student_train_steps,
        eval_tasks_per_type=eval_tasks_per_type,
        seed=seed,
        log_every=max(1, num_meta_steps // 20),
        verbose=True
    )
    
    # Run comparison
    all_results = run_comparison(config, strategies)
    
    # Save results
    for strategy, results in all_results.items():
        result_path = os.path.join(output_dir, f"{strategy}_{timestamp}.json")
        results.save(result_path)
        print(f"Saved: {result_path}")
    
    # Save summary
    summary = {
        "timestamp": timestamp,
        "config": {
            "num_meta_steps": num_meta_steps,
            "student_train_steps": student_train_steps,
            "eval_tasks_per_type": eval_tasks_per_type,
            "seed": seed,
        },
        "results": {
            strategy: {
                "final_accuracy": r.final_accuracy,
                "total_teacher_reward": r.total_teacher_reward,
                "total_time": r.total_time,
            }
            for strategy, r in all_results.items()
        }
    }
    
    summary_path = os.path.join(output_dir, f"summary_{timestamp}.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved: {summary_path}")
    
    return all_results


def run_single_experiment(
    strategy: str = "ucb1",
    num_meta_steps: int = 200,
    student_train_steps: int = 256,
    eval_tasks_per_type: int = 10,
    seed: int = 42,
    output_path: Optional[str] = None
) -> MetaTrainingResults:
    """
    Run a single training experiment.
    
    Args:
        strategy: Teacher strategy
        num_meta_steps: Number of meta-training steps
        student_train_steps: PPO steps per meta-step
        eval_tasks_per_type: Tasks per type in eval set
        seed: Random seed
        output_path: Optional path to save results
    
    Returns:
        MetaTrainingResults
    """
    # Determine teacher strategy
    if strategy in ["random", "fixed"]:
        teacher_strategy = strategy
    else:
        teacher_strategy = BanditStrategy(strategy)
    
    config = MetaTrainingConfig(
        num_meta_steps=num_meta_steps,
        student_train_steps=student_train_steps,
        eval_tasks_per_type=eval_tasks_per_type,
        teacher_strategy=teacher_strategy,
        seed=seed,
        log_every=max(1, num_meta_steps // 20),
        verbose=True
    )
    
    trainer = MetaTrainer(config)
    results = trainer.train()
    
    if output_path:
        results.save(output_path)
        print(f"Results saved: {output_path}")
    
    return results


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="TeachRL Experiment Runner")
    
    parser.add_argument(
        "--mode",
        choices=["single", "compare"],
        default="compare",
        help="Run single strategy or compare multiple"
    )
    parser.add_argument(
        "--strategy",
        default="ucb1",
        help="Strategy for single mode (ucb1, thompson_sampling, epsilon_greedy, random, fixed)"
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=None,
        help="Strategies for compare mode"
    )
    parser.add_argument(
        "--meta-steps",
        type=int,
        default=200,
        help="Number of meta-training steps"
    )
    parser.add_argument(
        "--train-steps",
        type=int,
        default=256,
        help="PPO training steps per meta-step"
    )
    parser.add_argument(
        "--eval-tasks",
        type=int,
        default=10,
        help="Evaluation tasks per task type"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--output",
        default="results",
        help="Output directory or file path"
    )
    
    args = parser.parse_args()
    
    if args.mode == "single":
        run_single_experiment(
            strategy=args.strategy,
            num_meta_steps=args.meta_steps,
            student_train_steps=args.train_steps,
            eval_tasks_per_type=args.eval_tasks,
            seed=args.seed,
            output_path=args.output if args.output != "results" else None
        )
    else:
        run_full_experiment(
            output_dir=args.output,
            num_meta_steps=args.meta_steps,
            student_train_steps=args.train_steps,
            eval_tasks_per_type=args.eval_tasks,
            seed=args.seed,
            strategies=args.strategies
        )


if __name__ == "__main__":
    main()
