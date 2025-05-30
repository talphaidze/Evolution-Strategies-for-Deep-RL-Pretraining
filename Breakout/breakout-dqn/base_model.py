from abc import ABC, abstractmethod
from typing import Dict, Any, List
import numpy as np

class BaseModel(ABC):
    @abstractmethod
    def get_parameters(self) -> np.ndarray:
        """Extract model parameters as a single flattened array."""
        pass
    
    @abstractmethod
    def set_parameters(self, params: np.ndarray) -> None:
        """Set model parameters from a single flattened array."""
        pass
    
    @abstractmethod
    def evaluate(self, num_episodes: int) -> float:
        """Evaluate the model for given number of episodes."""
        pass
    
    @abstractmethod
    def save_checkpoint(self, path: str, generation: int, metrics: Dict[str, Any]) -> None:
        """Save a checkpoint of the model."""
        pass