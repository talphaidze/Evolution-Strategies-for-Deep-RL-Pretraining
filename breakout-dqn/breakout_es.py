import argparse
from datetime import datetime
import os
import numpy as np
from copy import deepcopy
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.dqn import DQN
from typing import Dict, Any

import wandb
import ale_py

from es import EvolutionStrategy
from breakout_dqn_model import BreakoutDQN

def main():
    # take a debug argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", default=False)
    args = parser.parse_args()
    
    # Initialize the Breakout environment
    env = make_atari_env("ALE/Breakout-v5", n_envs=1, seed=42)
    env = VecFrameStack(env, n_stack=4)
    
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
    }
    
    # Initialize the Breakout model
    breakout_dqn_model = BreakoutDQN(env, dqn_config)
    
    es_config = {
        "population_size": 50,
        "sigma": 0.2,
        "learning_rate": 0.01,
        "num_episodes": 5,
        "save_freq": 10,
        "checkpoint_dir": models_dir,
    }
    
    # Initialize Evolution Strategy with the model
    es = EvolutionStrategy(
        model=breakout_dqn_model,
        population_size=es_config["population_size"],
        sigma=es_config["sigma"],
        learning_rate=es_config["learning_rate"],
        num_episodes=es_config["num_episodes"],
        save_freq=es_config["save_freq"],
        checkpoint_dir=es_config["checkpoint_dir"],
        debug=args.debug,
    )
    
    combined_config = {
        **dqn_config,
        **es_config
    }
    
    # Initialize wandb
    wandb.init(
        project="breakout-dqn-es",
        name=f"es_{current_datetime}",
        config=combined_config
    )
    
    # Train using evolution strategy
    es.train(num_generations=1000)

    wandb.finish()

if __name__ == "__main__":
    main() 