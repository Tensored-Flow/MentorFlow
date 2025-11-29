"""
PPO Student Agent for TeachRL
Wrapper around Stable-Baselines3 PPO for training the coding task student.
"""

import numpy as np
from typing import Optional, List, Dict, Any, Callable
from pathlib import Path

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import DummyVecEnv
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False
    print("Warning: stable-baselines3 not installed. Install with: pip install stable-baselines3")

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from student.student_env import StudentEnv, MultiTaskStudentEnv, make_env
from tasks.task_generator import TaskSpec, generate_eval_dataset, OBS_VEC_SIZE


class StudentAgent:
    """
    PPO-based student agent that learns to solve coding microtasks.
    """
    
    def __init__(
        self,
        learning_rate: float = 3e-4,
        n_steps: int = 128,
        batch_size: int = 64,
        n_epochs: int = 4,
        gamma: float = 0.99,
        seed: Optional[int] = None,
        verbose: int = 0,
        device: str = "auto"
    ):
        """
        Initialize student agent.
        
        Args:
            learning_rate: PPO learning rate
            n_steps: Steps per rollout
            batch_size: Minibatch size
            n_epochs: PPO epochs per update
            gamma: Discount factor
            seed: Random seed
            verbose: Verbosity level
            device: 'cpu', 'cuda', or 'auto'
        """
        if not HAS_SB3:
            raise ImportError("stable-baselines3 required. Install with: pip install stable-baselines3")
        
        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.seed = seed
        self.verbose = verbose
        self.device = device
        
        # Create default multi-task environment
        self._env = DummyVecEnv([lambda: MultiTaskStudentEnv(seed=seed)])
        
        # Initialize PPO model
        self.model = PPO(
            "MlpPolicy",
            self._env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            seed=seed,
            verbose=verbose,
            device=device
        )
        
        # Training history
        self.train_steps = 0
        self.eval_history: List[float] = []
    
    def train_on_task(self, arm_id: int, total_timesteps: int = 256) -> Dict[str, float]:
        """
        Train student on a specific task type.
        
        Args:
            arm_id: Task type (0-14)
            total_timesteps: Number of environment steps
        
        Returns:
            Training statistics
        """
        # Create environment for this task type
        env = DummyVecEnv([lambda aid=arm_id: make_env(arm_id=aid, seed=self.seed)])
        self.model.set_env(env)
        
        # Train
        self.model.learn(total_timesteps=total_timesteps, reset_num_timesteps=False)
        self.train_steps += total_timesteps
        
        return {"timesteps": total_timesteps, "total_steps": self.train_steps}
    
    def train_multi_task(self, total_timesteps: int = 1024) -> Dict[str, float]:
        """
        Train on random mix of all task types.
        
        Args:
            total_timesteps: Number of environment steps
        
        Returns:
            Training statistics
        """
        self.model.set_env(self._env)
        self.model.learn(total_timesteps=total_timesteps, reset_num_timesteps=False)
        self.train_steps += total_timesteps
        
        return {"timesteps": total_timesteps, "total_steps": self.train_steps}
    
    def predict(self, obs: np.ndarray, deterministic: bool = True) -> int:
        """
        Predict action for a single observation.
        
        Args:
            obs: Task observation vector
            deterministic: Use deterministic policy
        
        Returns:
            action (0-3)
        """
        if obs.ndim == 1:
            obs = obs.reshape(1, -1)
        action, _ = self.model.predict(obs, deterministic=deterministic)
        return int(action[0])
    
    def predict_batch(self, obs_batch: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """
        Predict actions for a batch of observations.
        
        Args:
            obs_batch: Batch of observations (N, OBS_VEC_SIZE)
            deterministic: Use deterministic policy
        
        Returns:
            actions array (N,)
        """
        actions, _ = self.model.predict(obs_batch, deterministic=deterministic)
        return actions
    
    def evaluate(self, eval_tasks: List[TaskSpec], deterministic: bool = True) -> float:
        """
        Evaluate student accuracy on a set of tasks.
        
        Args:
            eval_tasks: List of TaskSpec to evaluate on
            deterministic: Use deterministic policy
        
        Returns:
            Accuracy (0.0 to 1.0)
        """
        if not eval_tasks:
            return 0.0
        
        correct = 0
        for task in eval_tasks:
            obs = np.array(task.obs_vec, dtype=np.float32)
            action = self.predict(obs, deterministic=deterministic)
            if action == task.correct_action:
                correct += 1
        
        accuracy = correct / len(eval_tasks)
        self.eval_history.append(accuracy)
        return accuracy
    
    def evaluate_by_type(
        self,
        eval_tasks: List[TaskSpec],
        deterministic: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate accuracy broken down by task type.
        
        Returns:
            Dict mapping task type name to accuracy
        """
        from tasks.task_generator import arm_to_name, NUM_FAMILIES, NUM_DIFFICULTIES
        
        # Group tasks by type
        type_tasks: Dict[int, List[TaskSpec]] = {i: [] for i in range(15)}
        for task in eval_tasks:
            arm_id = task.family_id * NUM_DIFFICULTIES + task.difficulty_id
            type_tasks[arm_id].append(task)
        
        # Evaluate each type
        results = {}
        for arm_id, tasks in type_tasks.items():
            if tasks:
                correct = sum(
                    1 for t in tasks
                    if self.predict(np.array(t.obs_vec, dtype=np.float32), deterministic) == t.correct_action
                )
                results[arm_to_name(arm_id)] = correct / len(tasks)
            else:
                results[arm_to_name(arm_id)] = 0.0
        
        return results
    
    def save(self, path: str):
        """Save model to file."""
        self.model.save(path)
    
    def load(self, path: str):
        """Load model from file."""
        self.model = PPO.load(path, env=self._env, device=self.device)
    
    def reset(self):
        """Reset agent to untrained state."""
        self._env = DummyVecEnv([lambda: MultiTaskStudentEnv(seed=self.seed)])
        self.model = PPO(
            "MlpPolicy",
            self._env,
            learning_rate=self.learning_rate,
            n_steps=self.n_steps,
            batch_size=self.batch_size,
            n_epochs=self.n_epochs,
            gamma=self.gamma,
            seed=self.seed,
            verbose=self.verbose,
            device=self.device
        )
        self.train_steps = 0
        self.eval_history = []


class EvalCallback(BaseCallback):
    """Callback for periodic evaluation during training."""
    
    def __init__(
        self,
        eval_tasks: List[TaskSpec],
        eval_freq: int = 500,
        verbose: int = 0
    ):
        super().__init__(verbose)
        self.eval_tasks = eval_tasks
        self.eval_freq = eval_freq
        self.eval_results: List[float] = []
    
    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            # Evaluate
            correct = 0
            for task in self.eval_tasks:
                obs = np.array(task.obs_vec, dtype=np.float32).reshape(1, -1)
                action, _ = self.model.predict(obs, deterministic=True)
                if action[0] == task.correct_action:
                    correct += 1
            
            accuracy = correct / len(self.eval_tasks)
            self.eval_results.append(accuracy)
            
            if self.verbose:
                print(f"Step {self.n_calls}: accuracy={accuracy:.2%}")
        
        return True


# =============================================================================
# STANDALONE TEST
# =============================================================================

if __name__ == "__main__":
    if not HAS_SB3:
        print("Cannot run test: stable-baselines3 not installed")
        print("Install with: pip install stable-baselines3")
        exit(1)
    
    print("=" * 60)
    print("TeachRL PPO Student Agent Test")
    print("=" * 60)
    
    # Create evaluation dataset
    print("\nCreating evaluation dataset...")
    eval_tasks = generate_eval_dataset(tasks_per_type=5, seed=42)
    print(f"Eval dataset size: {len(eval_tasks)}")
    
    # Create student
    print("\nInitializing student agent...")
    student = StudentAgent(
        learning_rate=3e-4,
        n_steps=64,
        batch_size=32,
        seed=42,
        verbose=0
    )
    
    # Evaluate before training
    acc_before = student.evaluate(eval_tasks)
    print(f"Accuracy before training: {acc_before:.2%}")
    
    # Train on a specific task type
    print("\nTraining on arm_id=0 (var_trace_easy) for 512 steps...")
    student.train_on_task(arm_id=0, total_timesteps=512)
    
    # Evaluate after training
    acc_after = student.evaluate(eval_tasks)
    print(f"Accuracy after training: {acc_after:.2%}")
    print(f"Improvement: {acc_after - acc_before:+.2%}")
    
    # Train on multi-task
    print("\nTraining on multi-task for 512 more steps...")
    student.train_multi_task(total_timesteps=512)
    
    acc_final = student.evaluate(eval_tasks)
    print(f"Final accuracy: {acc_final:.2%}")
    
    # Evaluate by type
    print("\nAccuracy by task type:")
    type_acc = student.evaluate_by_type(eval_tasks)
    for name, acc in sorted(type_acc.items()):
        print(f"  {name}: {acc:.0%}")
    
    print(f"\nTotal training steps: {student.train_steps}")
    print("\n✓ PPO Student Agent test complete!")
