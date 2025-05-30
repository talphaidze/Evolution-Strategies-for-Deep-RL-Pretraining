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
│   │   ├── base_model.py
│   │   ├── breakout_dqn_model.py
│   │   ├── breakout_es.py
│   │   ├── breakout_sb3.py
│   │   ├── callbacks.py
│   │   ├── es.py
│   │   └── eval.py
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
├── MuJoCo/
│   └── [MuJoCo implementation files]
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
- **Type**: Continuous control tasks (HalfCheetah, Hopper, Walker2d)
- **Algorithms**: PPO vs ES
- **Architecture**: 4 hidden layers with 32 units each
- **Key Result**: PPO shows inconsistent performance across seeds but faster convergence when successful; ES more stable but significantly slower

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
cd MuJoCo/
# Follow similar pattern with respective training scripts
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
- **ES**: Lightweight architecture for efficient optimization

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
| MuJoCo | Stable but slow | Fast but unstable | ❌ No improvement |

## Authors

- Adrian Martínez López
- Ananya Gupta  
- Hanka Goralija
- Mario Rico Ibáñez
- Saúl Fenollosa Arguedas
- Tamar Alphaidze

*École Polytechnique Fédérale de Lausanne (EPFL)*
