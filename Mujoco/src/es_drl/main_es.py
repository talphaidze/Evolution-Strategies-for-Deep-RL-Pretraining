#!/usr/bin/env python

"""
Main script for training agents using Evolution Strategies (ES) and PPO.
This script handles command-line arguments, configuration loading, and initializes
the appropriate training algorithm (Basic ES, PPO, or Pretraining) based on the
provided configuration.
"""

import os

# Force MuJoCo to use EGL offscreen rendering (no GLX/GLFW)
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import argparse
import yaml

from src.es_drl.es.basic_es import BasicES
from src.es_drl.es.ppo import PPO
from src.es_drl.es.pretraining import Pretraining
from src.es_drl.es.tr_es import TrustRegionES


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="ES_DRL args")

    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to the config file'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )

    parser.add_argument(
        '--env_id',
        type=str,
        default="hopper",
        help='MuJoCo environment ID (all lowercase, e.g., halfcheetah)'
    )

    # common_cfg = load_yaml(sys.argv[1])
    args = parser.parse_args()
    es_cfg = load_yaml(args.config)
    seed = args.seed
    env_id = args.env_id

    if es_cfg["es_name"] == "basic_es":
        es = BasicES(es_cfg, seed=seed, env_id=env_id)
    elif es_cfg["es_name"] == "ppo":
        es = PPO(es_cfg, seed=seed, env_id=env_id)
    elif es_cfg["es_name"] == "pretraining":
        es = Pretraining(es_cfg, seed=seed, env_id=env_id)
    elif es_cfg["es_name"] == "tr_es":
        es = TrustRegionES(es_cfg, seed=seed, env_id=env_id)
    else:
        raise NotImplementedError(f"ES '{es_cfg['es_name']}' not implemented yet")

    ckpt = es.run()
    print(f"[ES] Training completed. Checkpoint saved at: {ckpt}")
