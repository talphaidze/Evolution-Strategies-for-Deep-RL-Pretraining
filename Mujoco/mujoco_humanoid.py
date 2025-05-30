import os

os.environ["MUJOCO_GL"] = "egl"  # headless EGL for MuJoCo
os.environ["PYOPENGL_PLATFORM"] = "egl"

import gymnasium as gym
import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.vec_env import (
    SubprocVecEnv,
    DummyVecEnv,
    VecVideoRecorder,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import NormalActionNoise

# ──────────────── CALLBACK ─────────────────
class VideoRecorderCallback(BaseCallback):
    """
    Record a video every `record_freq` env steps using a fresh
    single-env VecVideoRecorder and print the episode reward.
    """

    def __init__(self, record_freq, video_folder, video_length, env_id, verbose=0):
        super().__init__(verbose)
        self.record_freq = record_freq
        self.video_folder = video_folder
        self.video_length = video_length
        self.env_id = env_id
        self.last_record = 0

    def _on_step(self) -> bool:
        if (self.num_timesteps - self.last_record) >= self.record_freq:
            self.last_record = self.num_timesteps

            # create fresh single-env recorder
            record_env = DummyVecEnv(
                [lambda: Monitor(gym.make(self.env_id, render_mode="rgb_array"))]
            )
            record_env = VecVideoRecorder(
                record_env,
                video_folder=self.video_folder,
                record_video_trigger=lambda step: step == 0,
                video_length=self.video_length,
                # unique prefix so files don't overwrite
                name_prefix=f"td3-{self.env_id}-{self.num_timesteps}",
            )

            obs = record_env.reset()
            ep_reward = 0.0
            for _ in range(self.video_length):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, rewards, dones, _ = record_env.step(action)
                ep_reward += rewards[0]
                if dones[0]:
                    break
            record_env.close()

            print(
                f"[Video] step {self.num_timesteps:,}: "
                f"episode reward = {ep_reward:.1f}"
            )

        return True


# ───────────────────────────────────────────

if __name__ == "__main__":
    import multiprocessing as mp

    mp.set_start_method("spawn", force=True)

    ENV_ID = "Humanoid-v5"
    NUM_ENVS = 16
    SEED_BASE = 42

    def make_env(rank):
        def _init():
            env = gym.make(ENV_ID)
            env.reset(seed=SEED_BASE + rank)
            return env

        return _init

    train_env = SubprocVecEnv([make_env(i) for i in range(NUM_ENVS)])

    # Action-space size
    tmp_env = gym.make(ENV_ID)
    n_actions = tmp_env.action_space.shape[-1]
    tmp_env.close()

    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions)
    )

    model = TD3(
        "MlpPolicy",
        train_env,
        policy_kwargs=dict(net_arch=[400, 300]),
        learning_rate=3e-4,
        batch_size=256,
        buffer_size=1_000_000,
        learning_starts=10_000,
        train_freq=(1, "step"),
        gradient_steps=1,
        gamma=0.99,
        tau=0.005,
        target_policy_noise=0.2,
        target_noise_clip=0.5,
        action_noise=action_noise,
        verbose=1,
        seed=SEED_BASE,
    )

    # video every 1 000 steps, 1 000-step clip
    video_folder = "videos"
    os.makedirs(video_folder, exist_ok=True)
    video_callback = VideoRecorderCallback(
        record_freq=100_000,
        video_folder=video_folder,
        video_length=1_000,
        env_id=ENV_ID,
    )

    model.learn(total_timesteps=5_000_000, callback=video_callback)

    model.save("td3_humanoid")
    train_env.close()
