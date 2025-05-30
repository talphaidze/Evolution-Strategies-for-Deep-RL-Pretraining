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
Combined ES and PPO pretraining implementation.
This class implements a two-phase training approach where ES is used for initial
exploration and policy optimization, followed by PPO fine-tuning. It includes
parameter transfer between phases and support for wandb logging.
"""

from datetime import datetime
import os

from absl import logging
import wandb
import matplotlib.pyplot as plt
import jax
import imageio
from brax import envs
from brax.io import model, image

from src.es_drl.es import brax_training_utils
from src.es_drl.es import ppo_training_utils
from src.es_drl.es.base import EvolutionStrategy
from src.es_drl.utils.logger import Logger


class Pretraining(EvolutionStrategy):
    def __init__(self,  es_cfg: dict, seed: int, env_id: str):
        super().__init__(es_cfg, seed, env_id)
        # ES hyperparameters
        self.es_sigma = es_cfg["es_sigma"]
        self.es_population_size = es_cfg["es_population_size"]
        self.es_lr = es_cfg["es_learning_rate"]
        self.es_num_timesteps = es_cfg["es_num_timesteps"]
        self.episode_length = es_cfg.get("episode_length", 1000)

        # Network parameters
        self.hidden_sizes = es_cfg.get("hidden_sizes", [32, 32])

        # PPO hyperparameters
        self.ppo_num_envs = es_cfg["ppo_num_envs"]
        self.ppo_batch_size = es_cfg["ppo_batch_size"]
        self.ppo_lr = es_cfg["ppo_learning_rate"]
        self.ppo_num_timesteps = es_cfg["ppo_num_timesteps"]

        self.results_dir = f'results/pretraining/{self.env_id}'
        os.makedirs(self.results_dir, exist_ok=True)

        # Verbose flag
        self.verbose = es_cfg.get("verbose", False)

    def run(self) -> str:

        #Start a new wandb run to track this script.
        run = wandb.init(
            # Set the wandb entity where your project will be logged (generally your team name).
            entity="ES_DRL",
            # Set the wandb project where this run will be logged.
            project="ES_DRL Experiments",
            name=self.run_name,
            # Track hyperparameters and run metadata.
            config={
                "strategy": "pretraining",
                "seed": self.seed,
                "environment": self.env_id,
                "hidden_sizes": self.hidden_sizes,
                "episode_length": self.episode_length,
                "es_sigma": self.es_sigma,
                "es_population_size": self.es_population_size,
                "es_lr": self.es_lr,
                "es_num_timesteps": self.es_num_timesteps,
                "ppo_num_envs": self.ppo_num_envs,
                "ppo_batch_size": self.ppo_batch_size,
                "ppo_lr": self.ppo_lr,
                "ppo_num_timesteps": self.ppo_num_timesteps,
            },
        )

        xdata, ydata = [], []
        times = [datetime.now()]

        def progress(num_steps, metrics):
            times.append(datetime.now())
            xdata.append(num_steps)
            ydata.append(metrics["eval/episode_reward"])
            plt.xlim([0, self.es_num_timesteps + self.ppo_num_timesteps])
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

        logging.info("BEGINNING ES")
        prev_es_time = datetime.now()
        make_inference_fn, self.params, _ = brax_training_utils.train(
            environment=envs.get_environment(self.env_id),
            wrap_env=True,
            num_timesteps=self.es_num_timesteps,
            episode_length=self.episode_length,
            action_repeat=1,
            l2coeff=0,
            population_size=self.es_population_size,
            learning_rate=self.es_lr,
            fitness_shaping=brax_training_utils.FitnessShaping.WIERSTRA,
            num_eval_envs=128,
            perturbation_std=self.es_sigma,
            seed=self.seed,
            normalize_observations=True,
            num_evals=5,
            center_fitness=True,
            deterministic_eval=False,
            progress_fn=progress,
            hidden_layer_sizes=tuple(self.hidden_sizes),
        )
        wandb.log({"ES pretraining time": (datetime.now() - prev_es_time).seconds})


        logging.info("NOW STARTING PPO")
        prev_ppo_time = datetime.now()
        make_inference_fn, self.params, _ = ppo_training_utils.train(
            environment=envs.get_environment(self.env_id),
            num_timesteps=self.ppo_num_timesteps,
            num_evals=95,
            reward_scaling=0.1,
            episode_length=1000,
            normalize_observations=True,
            action_repeat=1,
            unroll_length=10,
            num_minibatches=32,
            num_updates_per_batch=8,
            discounting=0.97,
            learning_rate=self.ppo_lr,
            entropy_cost=1e-3,
            num_envs=self.ppo_num_envs,
            batch_size=self.ppo_batch_size,
            seed=self.seed,
            progress_fn=progress,
            restore_params=self.params,
            restore_value_fn=False
        )
        wandb.log({
            "PPO training time": (datetime.now() - prev_ppo_time),
        })

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
