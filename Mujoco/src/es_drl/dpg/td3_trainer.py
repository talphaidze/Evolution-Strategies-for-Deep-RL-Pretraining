# src/es_drl/dpg/td3_trainer.py
import os

# import torch TODO: CHANGE TO JAX
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


class VideoRecorderCallback(BaseCallback):
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
            record_env = DummyVecEnv(
                [lambda: Monitor(gym.make(self.env_id, render_mode="rgb_array"))]
            )
            record_env = VecVideoRecorder(
                record_env,
                video_folder=self.video_folder,
                record_video_trigger=lambda step: step == 0,
                video_length=self.video_length,
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
            print(f"[Video] step {self.num_timesteps:,}: reward = {ep_reward:.1f}")
        return True


class TD3Trainer:
    def __init__(self, common_cfg: dict, td3_cfg: dict, pretrained_path: str = None):
        # EGL for headless MuJoCo
        os.environ["MUJOCO_GL"] = "egl"
        os.environ["PYOPENGL_PLATFORM"] = "egl"

        # Load common settings
        self.env_id = common_cfg["env_id"]
        self.num_envs = common_cfg["num_envs"]
        self.seed = common_cfg["seed"]
        self.video_folder = common_cfg["video"]["folder_td3"]
        self.video_freq = common_cfg["video"]["freq_td3"]
        self.video_length = common_cfg["video"]["length"]
        self.total_timesteps = common_cfg["total_timesteps_td3"]

        # Prepare and vectorize the environment
        def make_env(rank):
            def _init():
                env = gym.make(self.env_id)
                env.reset(seed=self.seed + rank)
                from stable_baselines3.common.monitor import Monitor

                return Monitor(env, filename=None)  # logs per-episode info to infos

            return _init

        self.train_env = SubprocVecEnv([make_env(i) for i in range(self.num_envs)])

        # Prepare action noise
        tmp_env = gym.make(self.env_id)
        n_actions = tmp_env.action_space.shape[-1]
        tmp_env.close()
        action_noise = NormalActionNoise(
            mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions)
        )

        # Coerce learning_rate to a numeric constant if needed
        lr = td3_cfg["td3_kwargs"].get("learning_rate")
        if not isinstance(lr, (int, float)) and not callable(lr):
            try:
                td3_cfg["td3_kwargs"]["learning_rate"] = float(lr)
            except Exception:
                raise ValueError(f"Cannot convert learning_rate={lr!r} to float")

        # Convert train_freq list to tuple if necessary
        tf = td3_cfg["td3_kwargs"].get("train_freq")
        if isinstance(tf, list):
            td3_cfg["td3_kwargs"]["train_freq"] = tuple(tf)

        # Make sure video folder and log folder exist
        os.makedirs(self.video_folder, exist_ok=True)
        self.log_dir = os.path.join("logs", "td3", f"td3_seed{self.seed}")
        os.makedirs(self.log_dir, exist_ok=True)

        # Prepare episode logger
        from src.es_drl.utils.callbacks import EpisodeLoggerCallback
        from src.es_drl.utils.logger import Logger

        self.data_logger = Logger(self.log_dir)
        self.ep_callback = EpisodeLoggerCallback(self.data_logger)

        # Instantiate the TD3 model
        self.model = TD3(
            "MlpPolicy",
            self.train_env,
            **td3_cfg["td3_kwargs"],
            action_noise=action_noise,
            seed=self.seed,
        )

        # Load ES checkpoint into the TD3 actor (mu) only
        if pretrained_path is not None:
            es_state = torch.load(pretrained_path)
            # Get full policy state dict
            policy_state = self.model.policy.state_dict()

            # Map ES 'net.X.weight/bias' → 'actor.mu.X.weight/bias'
            loaded = 0
            for key, tensor in es_state.items():
                if key.startswith("net."):
                    subkey = key[len("net.") :]  # e.g. "0.weight"
                    actor_key = f"actor.mu.{subkey}"
                    if actor_key in policy_state:
                        policy_state[actor_key] = tensor
                        loaded += 1

            # Load updated policy (strict=False ignores missing keys)
            self.model.policy.load_state_dict(policy_state, strict=False)

            # Copy actor.mu → actor_target.mu
            actor_mu_state = self.model.policy.actor.mu.state_dict()
            self.model.policy.actor_target.mu.load_state_dict(
                actor_mu_state, strict=False
            )

            print(
                f"[TD3] Loaded {loaded} actor weights from ES checkpoint: {pretrained_path}"
            )

    def train(self):
        # Setup callbacks: video & episode logger
        video_cb = VideoRecorderCallback(
            record_freq=self.video_freq,
            video_folder=self.video_folder,
            video_length=self.video_length,
            env_id=self.env_id,
        )

        from stable_baselines3.common.callbacks import CallbackList

        callbacks = CallbackList([video_cb, self.ep_callback])

        # Start learning with both callbacks
        self.model.learn(total_timesteps=self.total_timesteps, callback=callbacks)

        # Save the final model
        save_dir = os.path.join("models", "td3")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"td3_finetuned_seed{self.seed}")
        self.model.save(save_path)
        print(f"[TD3] Training completed. Model saved at: {save_path}")

        self.train_env.close()
