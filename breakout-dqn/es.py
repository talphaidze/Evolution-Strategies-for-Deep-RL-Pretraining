import os
import numpy as np
import torch
import wandb
from datetime import datetime
from abc import ABC, abstractmethod
from typing import Dict, Any, List
import torch.multiprocessing as mp

def evaluate_individual(args):
    model_cls, theta, noise, sigma, num_episodes, gpu_id = args
    #CUDA uses the assigned GPU
    torch.cuda.set_device(gpu_id)
    #instantiate and move model to the correct device
    model = model_cls().to(f"cuda:{gpu_id}")
    perturbed_params = theta + sigma * noise
    model.set_parameters(perturbed_params)
    #evaluate on GPU
    reward = model.evaluate(num_episodes)
    return reward


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

class EvolutionStrategy:

    def __init__(
        self,
        model: BaseModel,
        population_size: int = 50,
        sigma: float = 0.1,
        learning_rate: float = 0.01,
        num_episodes: int = 5,
        save_freq: int = 10,
        checkpoint_dir: str = "es_checkpoints",
    ):
        """
        Initialize Evolution Strategy.
        
        Args:
            model: Model that implements BaseModel interface
            population_size: Number of individuals in population
            sigma: Standard deviation of noise
            learning_rate: Learning rate for parameter updates
            num_episodes: Number of episodes to evaluate each individual
            save_freq: How often to save checkpoints (in generations)
            checkpoint_dir: Directory to save checkpoints
        """
        self.model = model
        self.population_size = population_size
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.num_episodes = num_episodes
        self.save_freq = save_freq
        
        # Setup directories
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.run_dir = f"run_{timestamp}"
        self.checkpoint_dir = os.path.join(checkpoint_dir, self.run_dir)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
    
    def _evaluate_population(self, theta: np.ndarray, noises: List[np.ndarray]) -> List[float]:
        """Evaluate entire population using multiple GPUs."""
        num_gpus = torch.cuda.device_count()
        args = [
            (
                self.model.__class__,
                theta,
                noise,
                self.sigma,
                self.num_episodes,
                i % num_gpus  #assign GPUs
            )
            for i, noise in enumerate(noises)
        ]
        with mp.Pool(processes=self.population_size) as pool:
            rewards = pool.map(evaluate_individual, args)
        for i, reward in enumerate(rewards):
            print(f"Individual {i}: Reward = {reward:.2f}")
        return rewards


    def train(self, num_generations: int = 1000) -> None:
        """Train using evolution strategy."""
        # Get initial parameters
        theta = self.model.get_parameters()
        best_reward = float('-inf')
        
        for generation in range(num_generations):
            print(f"\nGeneration {generation}/{num_generations}")
            
            # Generate random noise for each member of the population
            noises = [np.random.normal(0, 1, theta.shape) for _ in range(self.population_size)]
            
            # Evaluate population
            rewards = self._evaluate_population(theta, noises)
            rewards = np.array(rewards)
            
            # Compute reward statistics
            mean_reward = np.mean(rewards)
            max_reward = np.max(rewards)
            
            # Update best reward and save checkpoint if needed
            if max_reward > best_reward:
                best_reward = max_reward
                if generation % self.save_freq == 0:
                    metrics = {
                        "reward": best_reward,
                        "generation": generation,
                        "mean_reward": mean_reward,
                        "max_reward": max_reward,
                    }
                    checkpoint_path = os.path.join(
                        self.checkpoint_dir, 
                        f"es_checkpoint_{generation}.pt"
                    )
                    self.model.save_checkpoint(checkpoint_path, generation, metrics)
            
            # Compute the reward-weighted sum of noise
            normalized_rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-8)
            weighted_sum = sum(r * n for r, n in zip(normalized_rewards, noises))
            
            # Update parameters
            theta = theta + self.learning_rate / (self.population_size * self.sigma) * weighted_sum
            
            # Log metrics
            wandb.log({
                "generation": generation,
                "mean_reward": mean_reward,
                "max_reward": max_reward,
                "best_reward": best_reward,
            })
            
            print(f"Generation {generation} stats:")
            print(f"Mean Reward: {mean_reward:.2f}")
            print(f"Max Reward: {max_reward:.2f}")
            print(f"Best Reward: {best_reward:.2f}")
        
        # Save final checkpoint
        final_metrics = {
            "reward": best_reward,
            "generation": num_generations,
            "mean_reward": mean_reward,
            "max_reward": max_reward,
        }
        final_path = os.path.join(self.checkpoint_dir, f"es_checkpoint_final.pt")
        self.model.save_checkpoint(final_path, num_generations, final_metrics)