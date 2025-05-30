# ES-DRL: Evolution Strategies with Deep Reinforcement Learning

This project implements a combination of Evolution Strategies (ES) and Deep Reinforcement Learning (DRL) for training agents in MuJoCo environments. The implementation includes multiple training strategies and supports various MuJoCo environments.

## Features

- Multiple training strategies:
  - Basic Evolution Strategies (ES)
  - Proximal Policy Optimization (PPO)
  - Pretraining (ES followed by PPO)
- Support for various MuJoCo environments
- Wandb integration for experiment tracking
- Video recording of agent behavior
- Comprehensive logging and early stopping

## Project Structure


```
src/es_drl/
├── es/
│   ├── base.py              # Base ES class with common functionality
│   ├── basic_es.py          # Basic ES implementation
│   ├── brax_training_utils.py # Training utilities for Brax environments
│   ├── ppo.py               # PPO implementation with clipped objective
│   ├── ppo_training_utils.py # PPO training utilities
│   └── pretraining.py       # Combined ES and PPO pretraining
├── utils/
│   ├── callbacks.py         # Training callbacks for monitoring
│   └── logger.py            # Logging utilities
└── main_es.py               # Main training script
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/es-drl.git
cd es-drl
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training with ES

```bash
python src/es_drl/main_es.py --config path/to/config.yaml --seed 42 --env_id hopper
```

### Training with PPO

```bash
python src/es_drl/main_es.py --config path/to/config.yaml --seed 42 --env_id hopper
```

### Pretraining (ES + PPO)

```bash
python src/es_drl/main_es.py --config path/to/config.yaml --seed 42 --env_id hopper
```

## Configuration

The training process is configured using YAML files. Here's an example configuration:

```yaml
# Common settings
es_name: "basic_es"  # or "ppo" or "pretraining"
num_timesteps: 1000000
episode_length: 1000
hidden_sizes: [400, 300]

# ES-specific settings
sigma: 0.1
population_size: 128
learning_rate: 1e-3

# PPO-specific settings
num_envs: 128
batch_size: 32
learning_rate: 1e-4
```

## Supported Environments

The following MuJoCo environments are supported:

- Ant
- HalfCheetah
- Hopper
- Humanoid
- HumanoidStandup
- Reacher
- Walker2d
- Pusher

## Training Process

### Evolution Strategies (ES)

The ES implementation follows the algorithm described in [Evolution Strategies as a Scalable Alternative to Reinforcement Learning](https://arxiv.org/pdf/1703.03864.pdf). The key components are:

1. Parameter perturbation:
   \[ \theta_i = \theta + \sigma \epsilon_i \]
   where $\epsilon_i \sim \mathcal{N}(0, I)$

2. Fitness evaluation:
   \[ F_i = \mathbb{E}[R(\theta_i)] \]

3. Update rule:
   \[ \theta_{t+1} = \theta_t + \alpha \frac{1}{n\sigma} \sum_{i=1}^n F_i \epsilon_i \]

### Proximal Policy Optimization (PPO)

The PPO implementation uses the clipped objective from [Proximal Policy Optimization Algorithms](https://arxiv.org/pdf/1707.06347.pdf):

\[ L^{CLIP}(\theta) = \hat{\mathbb{E}}_t[\min(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t)] \]

where:
- $r_t(\theta)$ is the probability ratio
- $\hat{A}_t$ is the advantage estimate
- $\epsilon$ is the clipping parameter

### Pretraining Strategy

The pretraining approach combines ES and PPO in two phases:

1. ES Phase:
   - Initial exploration using ES
   - Parameter optimization through fitness-based updates
   - Duration: $T_{ES}$ timesteps

2. PPO Phase:
   - Fine-tuning using PPO
   - Transfer of parameters from ES phase
   - Duration: $T_{PPO}$ timesteps

## Monitoring and Logging

### Wandb Integration

Training metrics are logged to Weights & Biases, including:
- Episode rewards
- Training time
- Parameter statistics
- Environment steps

### CSV Logging

Training progress is also saved to CSV files in the `logs` directory, containing:
- Step number
- Episode reward
- Training metrics

### Video Recording

Agent behavior can be recorded during training and evaluation:
```yaml
video:
  enabled: true
  frequency: 1000  # Record every 1000 episodes
  max_episodes: 10  # Maximum number of episodes to record
```

### Early Stopping

Training can be stopped early if the agent reaches a target reward threshold:
```yaml
early_stopping:
  enabled: true
  target_reward: 1000  # Stop when this reward is reached
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
