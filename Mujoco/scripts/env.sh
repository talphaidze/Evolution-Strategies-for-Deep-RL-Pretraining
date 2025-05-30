#!/usr/bin/env bash
# Minimal MuJoCo environment setup with JAX and utilities

conda create --name mujoco_environment python=3.10 -y
conda activate mujoco_environment

pip install -r requirements.txt

# TODO: Add torch if necessary
