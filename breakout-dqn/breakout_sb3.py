import os

from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.dqn import DQN

import ale_py  

env = make_atari_env(
    "ALE/Breakout-v5",
    n_envs=1,
    seed=42,
)
env = VecFrameStack(env, n_stack=4)
env.reset()


# os.chdir("Breakout")
models_dir = "DQN_models"
logdir = "DQN_logs"
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(logdir):
    os.makedirs(logdir)

# Train the agent
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

TIMESTEPS = 10000
iters = 0
for i in range(200):
    iters += 1
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="DQN_tb_log")
    model.save(f"{models_dir}/{TIMESTEPS*iters}")