# src/es_drl/es/base.py
"""
Base class for Evolution Strategies (ES) implementations.
This abstract class defines the common interface and functionality for all ES variants,
including directory management, configuration handling, and the abstract training loop.
"""

import os
from abc import ABC, abstractmethod


class EvolutionStrategy(ABC):
    def __init__(self, es_cfg: dict, seed: int, env_id: str):
        """
        es_cfg:   loaded from configs/es/<algo>.yaml
        seed:
        env_id:
        """
        self.es_cfg = es_cfg

        # Environment and seeds
        self.env_id = env_id
        self.seed = seed

        # Create directories for models, logs and videos
        self.es_name = es_cfg["es_name"]
        self.model_dir = os.path.join("models", "es", self.env_id, self.es_name)
        self.log_dir = os.path.join("logs", "es", self.es_name)
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        self.run_name = f"{self.env_id.upper()}-{self.es_name.upper()}-SEED={self.seed}"

    @abstractmethod
    def run(self) -> str:
        """
        Execute the ES training loop.
        Returns the path to the final checkpoint.
        """
        pass
