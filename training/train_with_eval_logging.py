"""Train PPO on a task with periodic evaluation and accuracy logging (multi-step aware)."""

import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO
from tqdm import tqdm

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from student.student_env import StudentEnv
from training.evaluate import evaluate_model
from training.callbacks import SharedProgressCallback


PPO_CFG = dict(
    learning_rate=3e-4,
    n_steps=256,
    batch_size=256,
    n_epochs=4,
    gamma=0.95,
    gae_lambda=0.92,
    clip_range=0.2,
    ent_coef=0.01,
    policy_kwargs={"net_arch": [128, 128, 64]},
)


@dataclass
class TrainingConfig:
    total_timesteps: int = 20_000
    train_steps_per_generation: int = 1_000  # Steps to train before each evaluation
    eval_problems: int = 100  # Number of problems to evaluate on each generation
    family_id: int = 0
    difficulty_id: int = 0
    save_dir: str = "models"
    model_name: str = "ppo_task_logged"
    verbose: int = 0


class TrainingLogger:
    def __init__(self, save_dir: str):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.accuracies: List[float] = []
        self.timesteps: List[int] = []
        
    def log_evaluation(self, timestep: int, accuracy: float) -> None:
        self.timesteps.append(timestep)
        self.accuracies.append(accuracy)
        
    def save_plot(self, filename: str = "training_accuracy.png") -> None:
        if not self.timesteps:
            print("Warning: No data to plot")
            return
        plt.figure(figsize=(10, 6))
        plt.plot(self.timesteps, self.accuracies, marker="o", linewidth=2, markersize=6)
        plt.xlabel("Timesteps", fontsize=12)
        plt.ylabel("Accuracy", fontsize=12)
        plt.title("PPO Training: Evaluation Accuracy Over Time", fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1.05)
        plt.tight_layout()
        save_path = self.save_dir / filename
        plt.savefig(save_path, dpi=150)
        print(f"Saved plot to {save_path}")
        plt.close()
        
    def save_metrics(self, filename: str = "training_metrics.npz") -> None:
        save_path = self.save_dir / filename
        np.savez(
            save_path,
            timesteps=np.array(self.timesteps),
            accuracies=np.array(self.accuracies),
        )
        print(f"Saved metrics to {save_path}")


def create_model(env: StudentEnv, config: TrainingConfig) -> PPO:
    cfg = PPO_CFG.copy()
    cfg["verbose"] = config.verbose
    return PPO("MlpPolicy", env, **cfg)


def train_with_eval_logging(
    config: Optional[TrainingConfig] = None,
) -> Tuple[PPO, TrainingLogger]:
    """
    Train PPO with periodic evaluation.
    
    Training cycle:
    1. Train for train_steps_per_generation timesteps
    2. Evaluate on eval_problems problems (100 by default)
    3. Log accuracy
    4. Repeat until total_timesteps reached
    """
    if config is None:
        config = TrainingConfig()
    
    env = StudentEnv(family_id=config.family_id, difficulty_id=config.difficulty_id)
    model = create_model(env, config)
    logger = TrainingLogger(config.save_dir)
    
    num_generations = config.total_timesteps // config.train_steps_per_generation
    desc = f"Training Task {config.family_id} (Diff {config.difficulty_id})"
    
    print(f"Training plan:")
    print(f"  - Total timesteps: {config.total_timesteps}")
    print(f"  - Train steps per generation: {config.train_steps_per_generation}")
    print(f"  - Evaluation problems per generation: {config.eval_problems}")
    print(f"  - Number of generations: {num_generations}")
    print()
    
    with tqdm(total=config.total_timesteps, desc=desc, unit="step") as pbar:
        for generation in range(num_generations):
            # Train for this generation
            progress_cb = SharedProgressCallback(pbar)
            model.learn(
                config.train_steps_per_generation,
                reset_num_timesteps=False,
                callback=progress_cb
            )
            
            current_timestep = model.num_timesteps
            
            # Evaluate on 100 problems
            pbar.write(f"\n[Generation {generation + 1}] Evaluating on {config.eval_problems} problems...")
            acc = evaluate_model(
                model,
                episodes=config.eval_problems,
                family_id=config.family_id,
                difficulty_id=config.difficulty_id
            )
            
            logger.log_evaluation(current_timestep, acc)
            pbar.write(f"[Generation {generation + 1}] Timestep: {current_timestep:6d}, Accuracy: {acc:.3f} ({acc*100:.1f}%)")
    
    model_path = logger.save_dir / f"{config.model_name}.zip"
    model.save(str(model_path))
    print(f"\nSaved trained model to {model_path}")
    
    logger.save_plot()
    logger.save_metrics()
    summary_path = logger.save_dir / f"{config.model_name}_summary.txt"
    if logger.accuracies:
        with open(summary_path, "w") as f:
            f.write("Train with eval logging summary\n")
            f.write(f"total_timesteps: {config.total_timesteps}\n")
            f.write(f"train_steps_per_generation: {config.train_steps_per_generation}\n")
            f.write(f"eval_problems: {config.eval_problems}\n")
            f.write(f"num_generations: {len(logger.accuracies)}\n")
            f.write(f"final_accuracy: {logger.accuracies[-1]:.4f}\n")
            f.write(f"max_accuracy: {max(logger.accuracies):.4f}\n")
    print(f"Saved summary to {summary_path}")
    
    return model, logger


def main():
    config = TrainingConfig(
        total_timesteps=20_000,
        train_steps_per_generation=1_000,
        eval_problems=100,  # Evaluate on 100 problems each generation
        family_id=0,
        difficulty_id=0,
    )
    
    model, logger = train_with_eval_logging(config)
    
    if logger.accuracies:
        print("\n" + "="*50)
        print("Training Summary:")
        print(f"  Final Accuracy:    {logger.accuracies[-1]:.3f}")
        print(f"  Max Accuracy:      {max(logger.accuracies):.3f}")
        print(f"  Mean Accuracy:     {np.mean(logger.accuracies):.3f}")
        print(f"  Accuracy Std Dev:  {np.std(logger.accuracies):.3f}")
        print("="*50)


if __name__ == "__main__":
    main()
