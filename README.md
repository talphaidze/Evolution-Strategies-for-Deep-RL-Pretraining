# Evolution Strategies for Deep RL Pretraining

This repository contains the implementation and experiments for comparing Evolution Strategies (ES) and Deep Reinforcement Learning (DRL) algorithms across environments of varying complexity, as well as investigating ES as a pretraining strategy for DRL.

## Abstract

Deep Reinforcement Learning has shown remarkable success in complex sequential decision-making tasks, but often requires extensive training and careful tuning. Evolution Strategies offer a simpler, gradient-free alternative that is computationally cheaper and easier to implement. This work compares ES and DRL performance across environments of varying complexity and analyzes the trade-offs between these approaches, with a focus on using ES as a pretraining strategy for DRL algorithms.

## Key Findings

- **ES is not consistently faster than DRL** and its effectiveness as a pretraining method is limited to simpler environments
- **ES provides limited benefit** in training speed and robustness to hyperparameter selection in complex tasks
- **In simple environments** like Flappy Bird, ES pretraining can accelerate DRL learning curves
- **In complex environments** like Breakout or MuJoCo, ES pretraining shows little to no benefit

## Repository Structure

```
├── Breakout
│   ├── breakout-dqn
│   │   ├── base_model.py # Base model interface
│   │   ├── breakout_dqn_model.py # BreakoutDQN model definition
│   │   ├── breakout_es.py # Main ES training script for Breakout
│   │   ├── breakout_sb3.py # Main DQN training script for Breakout
│   │   ├── callbacks.py # Custom callbacks for WandB logging
│   │   ├── es.py # Modular ES algorithm implementation
│   │   └── eval.py # Evaluation script for loading checkpoints
│   ├── requirements.txt
│   ├── run.batch
│   └── run.sh
├── FlappyBird
│   ├── es_dqn_flappy.mp4
│   ├── flappy_gym_env.py
│   ├── plot_dqn_es.py
│   ├── plot_es.py
│   ├── run_dqn.py
│   ├── run_es.py
│   ├── train_dqn.py
│   ├── train_dqn_es.py
│   └── train_es.py
├── Mujoco/
│   ├── src/
│   │   └── es_drl/
│   │       ├── es/
│   │       │   ├── base.py              # Base ES class with common functionality
│   │       │   ├── basic_es.py          # Basic ES implementation
│   │       │   ├── brax_training_utils.py # Training utilities for Brax environments
│   │       │   ├── ppo.py               # PPO implementation with clipped objective
│   │       │   ├── ppo_training_utils.py # PPO training utilities
│   │       │   └── pretraining.py       # Combined ES and PPO pretraining
│   │       ├── utils/
│   │       │   ├── callbacks.py         # Training callbacks for monitoring
│   │       │   └── logger.py            # Logging utilities
│   │       └── main_es.py               # Main training script
│   └── README.md
└── README.md

```

## Environments

### 1. Flappy Bird
- **Type**: Discrete action space, arcade-style game
- **Algorithms**: DQN vs ES
- **Architecture**: Fully connected MLP with 2 hidden layers (64 units each)
- **Key Result**: ES provides effective pretraining for DQN in this simple environment

### 2. Breakout
- **Type**: Discrete action space, Atari environment
- **Configurations**: 
  - Image-based input (ALE/Breakout-v5) with CNN
  - RAM-based input (Breakout-ram-v4) with MLP
- **Algorithms**: DQN vs ES
- **Key Result**: ES struggles with high-dimensional input spaces; DQN maintains performance advantage

### 3. MuJoCo Environments
- **Type**: Continuous control tasks (HalfCheetah, Hopper, Walker2d, Humanoid, Reacher, Swimmer)
- **Algorithms**: PPO vs ES
- **Architecture**: 4 hidden layers with 32 units each
- **Key Result**: PPO shows inconsistent performance across seeds and environments - while it converges 20x faster than ES in HalfCheetah, it fails to converge in Walker2d and Hopper. ES is slower but provides more stable and repeatable outcomes across all environments.

## Getting Started

### Prerequisites
```bash
pip install stable-baselines3>=2.0.0
pip install ale-py>=0.8.1
pip install "gymnasium[atari]>=0.28.1"
pip install torch>=2.0.0
pip install numpy>=1.21.0
pip install opencv-python>=4.5.0
pip install tensorboard>=2.14.0
pip install wandb
pip install pygame
pip install ple  # PyGame Learning Environment (for Flappy Bird)
pip install matplotlib
pip install brax  # For MuJoCo environments
pip install jax[cuda]  # For GPU support
pip install imageio[ffmpeg]  # For video recording
pip install mujoco  # MuJoCo physics engine
pip install mujoco_mjx  # MuJoCo MJX implementation
pip install joblib  # For parallel processing
pip install pre-commit  # For code quality checks
```
### Running Experiments

#### Breakout

**Training with ES:**
```bash
cd Breakout/breakout-dqn/
python breakout_es.py
```

**Training with DQN:**
```bash
cd Breakout/breakout-dqn/
python breakout_sb3.py
```

#### Flappy Bird

**Training with DQN:**
```bash
cd Flappy_Bird/
python train_dqn.py
```

**Training with ES:**
```bash
cd Flappy_Bird/
python train_es.py
```

**Training DQN with ES pretrained:**
```bash
cd Flappy_Bird/
python train_dqn_es.py
```

**Run DQN/ES policy:**
```bash
cd Flappy_Bird/
python run_dqn.py
```

#### MuJoCo
```bash
cd Mujoco/
python src/es_drl/main_es.py --config path/to/config.yaml --seed 42 --env_id hopper
```

## Algorithm Implementations

### Evolution Strategies (ES)
- **Population-based optimization** using Gaussian perturbations
- **Gradient estimation** via Monte Carlo sampling
- **Communication efficient** - only seeds and rewards exchanged
- **Naturally parallelizable** across multiple workers

### Deep Q-Networks (DQN)
- **Experience replay** for stable learning
- **Target networks** for improved stability
- **ε-greedy exploration** with linear decay

### Proximal Policy Optimization (PPO)
- **Actor-critic architecture** with separate value and policy networks
- **Clipped surrogate objective** for stable policy updates
- **Default Brax configurations** for MuJoCo environments

## Key Parameters

### Breakout
- **DQN**: Learning rate 2e-4, buffer size 100k, batch size 32
- **ES**: Population size 50, noise std 0.2, learning rate 0.01, 500 generations

### Flappy Bird
- **DQN**: Learning rate 5e-5, 1M timesteps, 8 parallel environments
- **ES**: Population size 16, noise std 0.05, 1000 generations

### MuJoCo
- **PPO**: Default Brax configurations with 8192 parallel environments
- **ES**: Population size 4096, noise std 0.01, learning rate 0.01, lightweight architecture for efficient optimization

## Logging and Monitoring

The repository uses **Weights & Biases (WandB)** for experiment tracking:
- Training curves and metrics
- Hyperparameter logging
- Model checkpointing
- Performance comparisons

## Results Summary

| Environment | ES Performance | DRL Performance | Pretraining Benefit |
|-------------|---------------|-----------------|-------------------|
| Flappy Bird | Good stability | Higher final reward | ✅ Effective |
| Breakout | Poor scaling | Strong performance | ❌ Limited |
| MuJoCo | Stable but slow | Fast but inconsistent | ❌ No improvement |

## Authors

- Adrian Martínez López
- Ananya Gupta  
- Hanka Goralija
- Mario Rico Ibáñez
- Saúl Fenollosa Arguedas
- Tamar Alphaidze

*École Polytechnique Fédérale de Lausanne (EPFL)*
