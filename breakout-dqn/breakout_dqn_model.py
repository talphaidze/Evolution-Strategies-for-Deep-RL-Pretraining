import os
import numpy as np
import torch
from copy import deepcopy
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.dqn import DQN
from typing import Dict, Any

import wandb
import ale_py

from es import BaseModel


class BreakoutDQN(BaseModel):
    def __init__(self, env, config):
        # Initialize the env and DQN model
        self.env = env
        self.model = DQN(
            "CnnPolicy",
            env,
            learning_rate=config["learning_rate"],
            buffer_size=config["buffer_size"],
            batch_size=config["batch_size"],
            learning_starts=config["learning_starts"],
            target_update_interval=config["target_update_interval"],
            train_freq=config["train_freq"],
            gradient_steps=config["gradient_steps"],
            exploration_fraction=config["exploration_fraction"],
            exploration_final_eps=config["exploration_final_eps"],
            optimize_memory_usage=config["optimize_memory_usage"],
            verbose=config["verbose"],
            tensorboard_log=config["tensorboard_log"],
        )
    
    def get_parameters(self) -> np.ndarray:
        """Extract model parameters as a single flattened array."""
        params = []
        for param in self.model.policy.parameters():
            params.append(param.data.view(-1).cpu().numpy())
        return np.concatenate(params)
    
    def set_parameters(self, params: np.ndarray) -> None:
        """Set model parameters from a single flattened array."""
        start = 0
        for param in self.model.policy.parameters():
            size = param.numel()
            param.data.copy_(torch.from_numpy(
                params[start:start + size]).view(param.size()))
            start += size
    
    def evaluate(self, num_episodes: int) -> float:
        """Evaluate the model for given number of episodes."""
        total_rewards = []
        
        for _ in range(num_episodes):
            obs = self.env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, _ = self.env.step(action)
                episode_reward += reward[0]
                
                if done:
                    break
            
            total_rewards.append(episode_reward)
        
        return np.mean(total_rewards)
    
    def save_checkpoint(self, path: str, generation: int, metrics: Dict[str, Any]) -> None:
        """Save a checkpoint of the model."""
        checkpoint = {
            "generation": generation,
            "model_state_dict": self.model.policy.state_dict(),
            "metrics": metrics,
        }
        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")