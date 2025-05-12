import os
import sys

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.dqn import DQN

import ale_py  
import wandb

from callbacks.wandb import WandbCallback

# Initialize wandb
wandb.init(project="breakout-dqn", name="breakout-dqn-sb3")

# Initialize environment
env = make_atari_env(
    "ALE/Breakout-v5",
    n_envs=1,
    seed=42,
)

# Stack frames
env = VecFrameStack(env, n_stack=4)
env.reset()

# Initialize models and logs directories
models_dir = "DQN_models"
logdir = "DQN_logs"
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(logdir):
    os.makedirs(logdir)

# Initialize model
model = DQN(
        "CnnPolicy",
        env,
        learning_rate=2e-4,
        buffer_size=100_000,
        batch_size=32,
        learning_starts=100_000,
        target_update_interval=5000,
        train_freq=4,
        gradient_steps=1,
        exploration_fraction=0.1,
        exploration_final_eps=0.01,
        optimize_memory_usage=False,
        verbose=1,
        tensorboard_log=logdir,
        )

# Train the agent
iters = 100
TIMESTEPS = 10000
for i in range(iters):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="DQN_tb_log", callback=WandbCallback())
    model.save(f"{models_dir}/{TIMESTEPS*i}")