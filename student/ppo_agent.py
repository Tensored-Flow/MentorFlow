"""PPO wrapper for student environments aligned with multi-step training."""

from typing import Optional, Tuple

from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm

from student.student_env import StudentEnv


PPO_DEFAULTS = dict(
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


class PPOAgent:
    def __init__(
        self,
        task_id: int = 0,
        difficulty: int = 0,
        model: Optional[BaseAlgorithm] = None,
        **ppo_kwargs,
    ):
        self.env = StudentEnv(family_id=task_id, difficulty_id=difficulty)
        cfg = PPO_DEFAULTS.copy()
        cfg.update(ppo_kwargs)
        if model is not None:
            self.model: BaseAlgorithm = model
            self.model.set_env(self.env)
        else:
            self.model = PPO("MlpPolicy", self.env, verbose=cfg.pop("verbose", 0), **cfg)

    def train(self, total_timesteps: int) -> None:
        self.model.learn(total_timesteps=total_timesteps)

    def predict(self, observation, deterministic: bool = True) -> Tuple[int, None]:
        return self.model.predict(observation, deterministic=deterministic)

    def save(self, path: str) -> None:
        self.model.save(path)

    @classmethod
    def load(cls, path: str, task_id: int = 0, difficulty: int = 0) -> "PPOAgent":
        env = StudentEnv(family_id=task_id, difficulty_id=difficulty)
        model = PPO.load(path, env=env)
        return cls(task_id=task_id, difficulty=difficulty, model=model)
