import argparse
from datetime import datetime
import os
import numpy as np
from copy import deepcopy
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from stable_baselines3.dqn import DQN
from typing import Dict, Any
import gymnasium as gym

import wandb
import ale_py

from callbacks import WandbCallback
from breakout_dqn_model import BreakoutDQN

def main():
    # take a debug argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--state", type=str, default="image")
    args = parser.parse_args()
    verbose = 1 if args.debug else 0
    
    # Initialize the Breakout environment
    if args.state == "image":
        env = make_atari_env("ALE/Breakout-v5", n_envs=1, seed=42)
        env = VecFrameStack(env, n_stack=4)
        
    elif args.state == "ram":
        # RAM state provides 128 bytes of the Atari 2600's RAM
        env = gym.make("Breakout-ram-v4", render_mode=None)
        env.reset(seed=42)
        env = DummyVecEnv([lambda: env])
    else:
        raise ValueError(f"Invalid state: {args.state}")
        return
    
    print("Environment created", flush=True)
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # Initialize models and logs directories
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    subdir = f"run_{current_datetime}"
    models_dir = os.path.join("DQN_sb3_models", subdir)
    logdir = os.path.join("DQN_sb3_logs", subdir)

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    
    dqn_config = {
        "policy": "MlpPolicy" if args.state == "ram" else "CnnPolicy",
        "learning_rate": 2e-4,
        "buffer_size": 100_000,
        "batch_size": 32,
        "learning_starts": 100_000,
        "target_update_interval": 1000,
        "train_freq": 4,
        "gradient_steps": 1,
        "exploration_fraction": 0.1,
        "exploration_final_eps": 0.01,
        "optimize_memory_usage": False,
        "verbose": verbose,
        "tensorboard_log": logdir,
    }
    
    sb3_config = {
        "iters": 100,
        "timesteps": 10000,
    }
    
    combined_config = {
        **dqn_config,
        **sb3_config
    }
    
    # Initialize the Breakout model
    breakout_dqn_model = BreakoutDQN(env, dqn_config)
    print(f"Number of parameters in DQN model: {sum(p.numel() for p in breakout_dqn_model.model.policy.parameters())}", flush=True)
    
    print("Models initialized", flush=True)
    
    # Initialize wandb
    wandb.init(
        project="breakout-dqn-sb3-{}".format(args.state),
        name=f"sb3_{current_datetime}",
        config=combined_config,
        mode="disabled" if args.debug else "online"
    )
    
    # Train dqn
    print("Training...", flush=True)
    iters = sb3_config["iters"]
    TIMESTEPS = sb3_config["timesteps"]
    for i in range(iters):
        breakout_dqn_model.model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="DQN_tb_log", callback=WandbCallback())
        breakout_dqn_model.model.save(f"{models_dir}/{TIMESTEPS*i}")
    print("Training complete", flush=True)

    wandb.finish()

if __name__ == "__main__":
    main() 