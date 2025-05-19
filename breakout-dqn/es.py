import os
import numpy as np
import torch
import wandb
from datetime import datetime
from abc import ABC, abstractmethod
from typing import Dict, Any, List
from base_model import BaseModel

class EvolutionStrategy:

    def __init__(
        self,
        model: BaseModel,
        num_generations: int,
        population_size: int = 50,
        sigma: float = 0.1,
        learning_rate: float = 0.01,
        num_episodes: int = 5,
        save_freq: int = 100,
        checkpoint_dir: str = "es_checkpoints",
        debug: bool = False,
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
        self.num_generations = num_generations
        self.population_size = population_size
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.num_episodes = num_episodes
        self.save_freq = save_freq
        self.debug = debug
        
        # Setup directories
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.run_dir = f"run_{timestamp}"
        self.checkpoint_dir = os.path.join(checkpoint_dir, self.run_dir)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
    
    def _evaluate_population(
        self, 
        theta: np.ndarray, 
        noises: List[np.ndarray],
        generation: int
    ) -> List[float]:
        """Evaluate entire population and return rewards."""
        rewards = []        
        for idx, noise in enumerate(noises):
            try:
                # Create perturbed parameters
                perturbed_params = theta + self.sigma * noise
                
                # Set parameters and evaluate
                self.model.set_parameters(perturbed_params)
                reward = self.model.evaluate(self.num_episodes)
                rewards.append(reward)
                
                if self.debug:
                    print(f"Generation {generation}, Individual {idx}: Reward = {reward:.2f}")
            except Exception as e:
                print(f"Error evaluating individual {idx}: {str(e)}")
                raise
            
        return rewards


    def train(self) -> None:
        # Get initial parameters
        theta = self.model.get_parameters()
        best_reward = float('-inf')
        
        for generation in range(self.num_generations):
            print(f"\nGeneration {generation}/{self.num_generations}", flush=True)
            
            # Generate random noise for each member of the population
            #noises = [np.random.normal(0, 1, theta.shape) for _ in range(self.population_size)]
            
            # Generate symmetric noise
            half_pop = self.population_size // 2
            noise_half = [np.random.normal(0, 1, theta.shape) for _ in range(half_pop)]
            noises = noise_half + [-n for n in noise_half]  # symmetric perturbations

            # Evaluate population
            rewards = self._evaluate_population(theta, noises, generation)
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

            # Ranking-based reward shaping
            # ranks = np.argsort(np.argsort(rewards))  # Rank rewards
            # shaped_rewards = (ranks - (self.population_size - 1) / 2) / ((self.population_size - 1) / 2)
            # shaped_rewards = shaped_rewards - np.mean(shaped_rewards)  # Mean-zero

            # # Weighted sum of noise
            # weighted_sum = sum(r * n for r, n in zip(shaped_rewards, noises))
            # theta = theta + self.learning_rate / (self.population_size * self.sigma) * weighted_sum

            # Log metrics
            wandb.log({
                "mean_reward": mean_reward,
                "max_reward": max_reward,
                "best_reward_so_far": best_reward,
            }, step=generation)
            
            print(f"Generation {generation} done")
            print(f"Mean Reward: {mean_reward:.2f}")
            print(f"Max Reward: {max_reward:.2f}")
            print(f"Best Reward so far: {best_reward:.2f}")
        
        # Save final checkpoint
        final_metrics = {
            "reward": best_reward,
            "generation": self.num_generations,
            "mean_reward": mean_reward,
            "max_reward": max_reward,
        }
        final_path = os.path.join(self.checkpoint_dir, f"es_checkpoint_final.pt")
        self.model.save_checkpoint(final_path, self.num_generations, final_metrics)