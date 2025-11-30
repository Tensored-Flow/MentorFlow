"""Frozen GPT-2 feature extractor for SB3 PPO."""

from typing import Any, Dict

import gymnasium as gym
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn
from transformers import AutoModel


class GPT2FeatureExtractor(BaseFeaturesExtractor):
    """Use a frozen GPT-2 model to embed tokenized text observations."""

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        model_name: str = "gpt2",
        max_length: int = 128,
    ):
        # observation_space shape: (max_length,)
        super().__init__(observation_space, features_dim=0)
        self.max_length = max_length
        self.gpt2 = AutoModel.from_pretrained(model_name)
        # Freeze GPT-2 parameters.
        for p in self.gpt2.parameters():
            p.requires_grad = False
        hidden_size = self.gpt2.config.hidden_size
        self.features_dim = hidden_size

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # observations: (batch, max_length) token ids.
        input_ids = observations.long()
        attention_mask = (input_ids != 0).long()
        outputs = self.gpt2(input_ids=input_ids, attention_mask=attention_mask)
        # Use the last hidden state of the last token (or mean pool).
        last_hidden = outputs.last_hidden_state  # (batch, seq, hidden)
        pooled = last_hidden[:, -1, :]
        return pooled
