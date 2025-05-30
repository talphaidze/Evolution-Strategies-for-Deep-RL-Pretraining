import time
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from collections import deque

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback, CallbackList

from flappy_gym_env import FlappyBirdEnv


def main():
    base_seed = 42
    n_envs    = 1
    vec_env = DummyVecEnv([
        (lambda i=i: FlappyBirdEnv(display_screen=False,
                                frame_skip=4,
                                seed=base_seed + i))
        for i in range(n_envs)
    ])
    vec_env = VecMonitor(vec_env)


    model = DQN(
        "MlpPolicy",
        vec_env,
        learning_rate=5e-5,
        buffer_size=10_000,
        learning_starts=1_000,
        batch_size=32,
        train_freq=(1, "step"),
        gradient_steps=16,
        target_update_interval=1_000,
        gamma=0.90,
        exploration_initial_eps=0.2,
        exploration_final_eps=0.0001,
        exploration_fraction=0.1,
        policy_kwargs={
            "net_arch": [64, 64],
            "activation_fn": nn.Tanh,
        },
        device="auto",
        tensorboard_log="./tensorboard/dqn/",
        verbose=1,
    )

    model.learn(total_timesteps=1_000_000)

    model.save("dqn_flappy")

if __name__ == '__main__':
    main()
