import gym
import torch
import numpy as np
import os
import random
import time
import pickle
from ple import PLE
from ple.games.flappybird import FlappyBird
from gym import spaces

from torch import nn
from torch.nn.utils.stateless import functional_call
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from gym.utils import seeding

from flappy_gym_env import FlappyBirdEnv

class MLP(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_sizes = (64, 64)):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.Tanh())
            in_dim = h
        layers.append(nn.Linear(in_dim, action_dim))
        layers.append(nn.Softmax(dim=-1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


@torch.no_grad()
def evaluate_candidate(flat_params: torch.Tensor, template: nn.Module, env: gym.Env, device: torch.device) -> float:

    pointer = 0
    state_dict = {}
    for name, param in template.named_parameters():
        numel = param.numel()
        chunk = flat_params[pointer:pointer+numel].view(param.shape).to(param.device, param.dtype)
        state_dict[name] = chunk
        pointer += numel

    env_obs = env.reset()
    total_r = 0.0
    done = False
    while not done:
        obs_t = torch.tensor(env_obs, dtype=torch.float32, device=device)
        probs = functional_call(template, state_dict, (obs_t,))
        action = torch.argmax(probs).item()
        env_obs, r, done, _ = env.step(action)
        total_r += r
    return total_r


def train_es():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # create and seed env
    env = FlappyBirdEnv(display_screen=False, seed=42)
    obs_dim    = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # build template policy
    template = MLP(obs_dim, action_dim, hidden_sizes=[64, 64]).to(device)
    flat0    = torch.nn.utils.parameters_to_vector(template.parameters()).detach()

    # ES hyperparameters
    num_envs    = 16
    sigma       = 0.05
    lr          = 0.005
    generations = 3000
    D = flat0.numel()

    # parameter matrix
    param_matrix = flat0.unsqueeze(0).repeat(num_envs, 1).to(device)

    folder = "es_policies"
    os.makedirs(folder, exist_ok=True)

    # TensorBoard writer
    writer = SummaryWriter(log_dir="./tensorboard/es/")
    max_r = float("-inf")
    best = None
    try:
        for gen in range(generations):
            noise = torch.randn(num_envs, D, device=device)
            half = num_envs // 2
            noise = torch.cat([noise[:half], -noise[:half]], dim=0)
            candidates = param_matrix + sigma * noise

            # evaluate candidates & fill buffer
            rewards = torch.zeros(num_envs, device=device)
            for i in range(num_envs):
                obs = env.reset()
                done = False
                while not done:
                    r = evaluate_candidate(candidates[i], template, env, device)
                    rewards[i] = r
                    done = True

            # ES update
            ranks  = torch.argsort(torch.argsort(rewards))
            shaped = (ranks.float() - (num_envs - 1) / 2) / ((num_envs - 1) / 2)
            shaped -= shaped.mean()
            grad = (noise.t() @ shaped) / (sigma * num_envs)
            param_matrix += lr * grad.unsqueeze(0)

            # log to TB
            writer.add_scalar("ES/MeanReward", rewards.mean().item(), gen)
            writer.add_scalar("ES/MaxReward",  rewards.max().item(),  gen)
            
            gen_best_idx    = torch.argmax(rewards).item()
            gen_best_reward = rewards[gen_best_idx].item()
            if rewards.mean().item() > max_r:
                max_r = rewards.mean().item()
                best  = candidates[gen_best_idx].cpu()

                fn = os.path.join(folder, f"best_policy_so_far_gen{gen}.pth")
                torch.save(best, fn)
                print(f"[ES] New best reward {max_r:.2f} at gen {gen}, saved policy.")

            # periodic saves as well
            if gen % 500 == 0 and gen > 0:
                torch.save(best, f"policy_gen{gen}.pth")

    finally:
        writer.close()

        env.close()



if __name__ == "__main__":
    train_es()

