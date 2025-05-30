# Copyright 2024 The Brax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Basic Evolution Strategy (ES) implementation.
This class implements the core ES algorithm with population-based optimization,
parameter perturbation, and fitness-based updates. It includes support for
wandb logging, video recording, and checkpointing.

See: https://arxiv.org/pdf/1703.03864.pdf
"""

from datetime import datetime
import os

import wandb
import matplotlib.pyplot as plt
import jax
import imageio
from brax import envs
from brax.io import model, image

from src.es_drl.es import brax_training_utils
from src.es_drl.es.base import EvolutionStrategy
from src.es_drl.utils.logger import Logger


class BasicES(EvolutionStrategy):
    def __init__(self,  es_cfg: dict, seed: int, env_id: str):
        super().__init__(es_cfg, seed, env_id)

        self.hidden_sizes = es_cfg.get("hidden_sizes", [400, 300])

        # ES hyperparameters
        self.sigma = es_cfg["sigma"]
        self.population_size = es_cfg["population_size"]
        self.lr = es_cfg["learning_rate"]
        self.num_timesteps = es_cfg["num_timesteps"]
        self.episode_length = es_cfg.get("episode_length", 1000)

        # Network parameters
        self.hidden_sizes = es_cfg.get("hidden_sizes", [32, 32])


        self.results_dir = f'results/es/{self.env_id}'
        os.makedirs(self.results_dir, exist_ok=True)

        # Verbose flag
        self.verbose = es_cfg.get("verbose", False)

    def _save_video(self):
        # create an env with auto-reset
        env = envs.create(env_name=self.env_id)

        jit_env_reset = jax.jit(env.reset)
        jit_env_step = jax.jit(env.step)
        jit_inference_fn = jax.jit(self.inference_fn)

        rollout = []
        rng = jax.random.PRNGKey(seed=1)
        state = jit_env_reset(rng=rng)
        for _ in range(self.episode_length):
            rollout.append(state.pipeline_state)
            act_rng, rng = jax.random.split(rng)
            act, _ = jit_inference_fn(state.obs, act_rng)
            state = jit_env_step(state, act)

        frames = image.render_array(
            env.sys, jax.device_get(rollout), height=480, width=640
        )
        fps = int(1.0 / env.dt)
        with imageio.get_writer(
            f"{self.results_dir}/{self.es_name}_seed{self.seed}.mp4", fps=fps
        ) as w:
            for frame in frames:
                w.append_data(frame)

        wandb.log(
            {
                "rollout_video": wandb.Video(
                    f"{self.results_dir}/{self.es_name}_seed{self.seed}.mp4",
                    fps=30,
                    format="mp4",
                )
            }
        )

    def run(self) -> str:

        # Start a new wandb run to track this script.
        run = wandb.init(
            # Set the wandb entity where your project will be logged (generally your team name).
            entity="ES_DRL",
            # Set the wandb project where this run will be logged.
            project="ES_DRL Experiments",
            name=self.run_name,
            # Track hyperparameters and run metadata.
            config={
                "strategy": "es",
                "seed": self.seed,
                "environment": self.env_id,
                "hidden_sizes": self.hidden_sizes,
                "sigma": self.sigma,
                "population_size": self.population_size,
                "lr": self.lr,
                "num_timesteps": self.num_timesteps,
                "episode_length": self.episode_length,
            },
        )

        xdata, ydata = [], []
        times = [datetime.now()]

        def progress(num_steps, metrics):
            times.append(datetime.now())
            xdata.append(num_steps)
            ydata.append(metrics["eval/episode_reward"])
            plt.xlim([0, self.num_timesteps])
            plt.ylim([min_y, max_y])
            plt.xlabel("# environment steps")
            plt.ylabel("reward per episode")
            plt.plot(xdata, ydata)
            print(f"Reward: {metrics['eval/episode_reward']}")
            metrics_to_log = {
                "Reward": metrics["eval/episode_reward"],
                "Step Time": (times[-1] - times[-2]).seconds,
                "Cumulative Time": (times[-1] - times[0]).seconds,
            }
            curr_reward = metrics_to_log["Reward"] 
            if not reached_rewards["reward_100"] and curr_reward >= max_y:
                metrics_to_log["Time to Max Reward"] = metrics_to_log["Cumulative Time"]
                reached_rewards["reward_100"] = True
            elif not reached_rewards["reward_75"] and curr_reward >= (0.75 * max_y):
                metrics_to_log["Time to 75% Reward"] = metrics_to_log["Cumulative Time"]
                reached_rewards["reward_75"] = True
            elif not reached_rewards["reward_50"] and curr_reward >= (0.5 * max_y):
                metrics_to_log["Time to 50% Reward"] = metrics_to_log["Cumulative Time"]
                reached_rewards["reward_50"] = True
            elif not reached_rewards["reward_25"] and curr_reward >= (0.25 * max_y):
                metrics_to_log["Time to 25% Reward"] = metrics_to_log["Cumulative Time"]
                reached_rewards["reward_25"] = True

            run.log(metrics_to_log)
            plt.savefig(f"{self.results_dir}/{self.es_name}_seed{self.seed}.png")
            return reached_rewards["reward_100"]

        reached_rewards = {
            "reward_25": False,
            "reward_50": False,
            "reward_75": False,
            "reward_100": False,
        }
        max_y = {
            "ant": 8000,
            "halfcheetah": 8000,
            "hopper": 2500,
            "humanoid": 13000,
            "humanoidstandup": 75_000,
            "reacher": 5,
            "walker2d": 5000,
            "pusher": 0,
        }[self.env_id]
        min_y = {"reacher": -100, "pusher": -150}.get(self.env_id, 0)

        make_inference_fn, self.params, _ = brax_training_utils.train(
            environment=envs.get_environment(self.env_id),
            wrap_env=True,
            num_timesteps=self.num_timesteps,
            episode_length=self.episode_length,
            action_repeat=1,
            l2coeff=0,
            population_size=self.population_size,
            learning_rate=self.lr,
            fitness_shaping=brax_training_utils.FitnessShaping.WIERSTRA,
            num_eval_envs=128,
            perturbation_std=self.sigma,
            seed=self.seed,
            normalize_observations=True,
            num_evals=100,
            center_fitness=True,
            deterministic_eval=False,
            progress_fn=progress,
            hidden_layer_sizes=tuple(self.hidden_sizes),
        )

        print(f"time to jit: {(times[1] - times[0]).seconds}")
        print(f"time to train: {(times[-1] - times[1]).seconds}")
        run.log(
            {
                "JIT-Time": (times[1] - times[0]).seconds,
                "Training Time": (times[-1] - times[1]).seconds,
            }
        )
        wandb.log(
            {
                "training_rewards": wandb.Image(
                    f"{self.results_dir}/{self.es_name}_seed{self.seed}.png"
                )
            }
        )

        self.inference_fn = make_inference_fn(self.params)

        model.save_params(
            os.path.join(self.model_dir, f"{self.es_name}_seed{self.seed}.pt"),
            self.params,
        )
        artifact = wandb.Artifact(
            name="model_params",
            type="model",
            description="Final Brax policy parameters",
        )
        artifact.add_file(
            os.path.join(self.model_dir, f"{self.es_name}_seed{self.seed}.pt"),
        )
        wandb.log_artifact(artifact)

        # self._save_video()
        run.finish()
