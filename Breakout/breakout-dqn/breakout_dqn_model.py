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

from base_model import BaseModel

class BreakoutDQN(BaseModel):
    def __init__(self, env, config):
        # Initialize the env and DQN model
        self.env = env
        self.model = DQN(
            config["policy"] if "policy" in config else "CnnPolicy",
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
            params.append(param.data.cpu().numpy().ravel())
        return np.concatenate(params)
    
    def set_parameters(self, params: np.ndarray) -> None:
        """Set model parameters from a single flattened array."""
        start = 0
        device = next(self.model.policy.parameters()).device
        
        for i, param in enumerate(self.model.policy.parameters()):
            size = param.numel()            
            try:
                param_slice = params[start:start + size]
                reshaped_params = param_slice.reshape(param.size())
                tensor_params = torch.from_numpy(reshaped_params).to(dtype=param.dtype, device=device)
                
                if torch.isnan(tensor_params).any():
                    print(f"Warning: NaN values in parameter {i}")
                if torch.isinf(tensor_params).any():
                    print(f"Warning: Inf values in parameter {i}")
                
                param.data.copy_(tensor_params)
            except Exception as e:
                print(f"Error setting parameter {i}: {str(e)}")
                print(f"Parameter slice shape: {param_slice.shape}")
                print(f"Expected shape: {param.size()}")
                raise
            
            start += size
    
    def evaluate(self, num_episodes: int) -> float:
        """Evaluate the model for given number of episodes."""
        total_rewards = []
        total_episode_lengths = []
        
        # Actions: 0=NOOP, 1=FIRE, 2=RIGHT, 3=LEFT
        FIRE_ACTION = 1
        
        for episode in range(num_episodes):
            obs = self.env.reset()
            done = False
            episode_reward = 0
            lives = 5
            episode_length = 0
            
            # Start the game by firing the ball
            obs, rewards, dones, _ = self.env.step([FIRE_ACTION])
            self.model.num_timesteps += 1
            episode_length += 1
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, rewards, dones, info = self.env.step(action)
                self.model.num_timesteps += 1
                episode_length += 1
                episode_reward += rewards[0]
                done = dones[0]

                # Fire to start new life if ball is lost
                if 'lives' in info[0] and info[0]['lives'] < lives:
                    lives = info[0]['lives']
                    obs, rewards, dones, _ = self.env.step([FIRE_ACTION])
                    self.model.num_timesteps += 1
                    episode_length += 1
                if done:
                    break
            
            total_rewards.append(episode_reward)
            total_episode_lengths.append(episode_length)
        mean_reward = np.mean(total_rewards)
        mean_episode_length = np.mean(total_episode_lengths)
        return mean_reward, mean_episode_length
    
    def save_checkpoint(self, path: str, generation: int, metrics: Dict[str, Any]) -> None:
        """Save a checkpoint of the model."""
        checkpoint = {
            "generation": generation,
            "model_state_dict": self.model.policy.state_dict(),
            "metrics": metrics,
        }
        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")