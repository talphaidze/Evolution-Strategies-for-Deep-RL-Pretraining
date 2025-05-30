import torch
import torch.nn as nn
from stable_baselines3.common.callbacks import BaseCallback

import numpy as np
import gym
from ple import PLE
from ple.games.flappybird import FlappyBird
from gym import spaces

from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3 import DQN

from train_es import MLP
from flappy_gym_env import FlappyBirdEnv

def unflatten_params(flat, template, cum_sizes):
    sd = {}
    for i, (name, p) in enumerate(template.named_parameters()):
        start = cum_sizes[i].item()
        end   = cum_sizes[i+1].item()
        sd[name] = flat[start:end].view(p.shape).to(p.device, p.dtype)
    return sd

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # env & dims
    base_seed = 42
    n_envs    = 1
    vec_env = DummyVecEnv([
        (lambda i=i: FlappyBirdEnv(display_screen=False,
                                frame_skip=4,
                                seed=base_seed + i))
        for i in range(n_envs)
    ])
    env = VecMonitor(vec_env)
    obs    = env.reset()
    obs_dim    = env.observation_space.shape[0]
    action_dim = env.action_space.n

    es_template = MLP(obs_dim, action_dim).to(device).double()
    numels = [p.numel() for p in es_template.parameters()]
    cum_sizes = torch.tensor([0] + list(torch.cumsum(torch.tensor(numels), 0)))

    # load your flat ES policy
    flat = torch.load("best_policy_so_far_gen1266.pth", map_location=device)
    es_sd = unflatten_params(flat, es_template, cum_sizes)

    # extract only hidden-layer weights (layers 0 & 2)
    hidden_keys = [ "net.0.weight","net.0.bias","net.2.weight","net.2.bias"]
    es_hidden = [es_sd[k] for k in hidden_keys]

    # create SB3 DQN with a 64-64 Tanh backbone
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=5e-5,
        buffer_size=10_000,
        learning_starts=1_000,
        batch_size=32,
        train_freq=(1, "step"),
        gradient_steps=16,
        target_update_interval=1_000,
        gamma=0.90,
        exploration_initial_eps=0.1,
        exploration_final_eps=0.0001,
        exploration_fraction=0.05,
        policy_kwargs={
            "net_arch": [64, 64],
            "activation_fn": nn.Tanh,
        },
        device="auto",
        tensorboard_log="./tensorboard/dqn_es/",
        verbose=1,
    )

    # inject ES hidden weights into model.policy.q_net
    qnet_params = list(model.policy.q_net.parameters())
    for i in range(4):
        qnet_params[i].data.copy_(es_hidden[i])

    n_warmup = 10000
    obs = env.reset()
    for _ in range(n_warmup):
        action, _ = model.predict(obs, deterministic=True)
        new_obs, reward, done, info = env.step(action)
        model.replay_buffer.add(obs, new_obs, action, reward, done, info)
        obs = new_obs

    model.learn(total_timesteps=1_000_000)

    model.save("dqn_es_flappy")

if __name__ == "__main__":
    main()
