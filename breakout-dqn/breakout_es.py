import argparse
from datetime import datetime
import os
import numpy as np
from copy import deepcopy
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from stable_baselines3.dqn import DQN
from typing import Dict, Any

import wandb
import ale_py

from es import EvolutionStrategy
from breakout_dqn_model import BreakoutDQN
import gymnasium as gym

def main():
    # take a debug argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--state", type=str, default="image")
    args = parser.parse_args()
    
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
    models_dir = os.path.join("DQN_es_models", subdir)
    logdir = os.path.join("DQN_es_logs", subdir)

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
        "target_update_interval": 5000,
        "train_freq": 4,
        "gradient_steps": 1,
        "exploration_fraction": 0.1,
        "exploration_final_eps": 0.01,
        "optimize_memory_usage": False,
        "verbose": 1,
        "tensorboard_log": logdir,
        "policy_kwargs": {
            "net_arch": [256, 256] if args.state == "ram" else None
        }
    }
    
    es_config = {
        "num_generations": 10,
        "population_size": 10,
        "sigma": 0.2,
        "learning_rate": 0.01,
        "num_episodes": 5,
        "save_freq": 100,
        "checkpoint_dir": models_dir,
    }
    
    combined_config = {
        **dqn_config,
        **es_config
    }
    
    # Initialize the Breakout model
    breakout_dqn_model = BreakoutDQN(env, dqn_config)
    print(f"Number of parameters in DQN model: {sum(p.numel() for p in breakout_dqn_model.model.policy.parameters())}", flush=True)
    
    # Initialize Evolution Strategy with the model
    es = EvolutionStrategy(
        model=breakout_dqn_model,
        num_generations=es_config["num_generations"],
        population_size=es_config["population_size"],
        sigma=es_config["sigma"],
        learning_rate=es_config["learning_rate"],
        num_episodes=es_config["num_episodes"],
        save_freq=es_config["save_freq"],
        checkpoint_dir=es_config["checkpoint_dir"],
        debug=args.debug,
    )
    print("Models initialized", flush=True)
    
    # Initialize wandb
    wandb.init(
        project="breakout-dqn-es-{}".format(args.state),
        name=f"es_{current_datetime}",
        config=combined_config,
        mode="disabled" if args.debug else "online"
    )
    
    # Train using evolution strategy
    print("Training...", flush=True)
    es.train()
    print("Training complete", flush=True)

    wandb.finish()

if __name__ == "__main__":
    main() 