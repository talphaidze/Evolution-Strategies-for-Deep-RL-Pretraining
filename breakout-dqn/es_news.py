import argparse
from datetime import datetime
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from multiprocessing import Pool, cpu_count
import pickle
from typing import Dict, Any, List, Tuple
import wandb
import ale_py

class BreakoutCNN(nn.Module):
    """CNN architecture for Breakout similar to DQN but for policy output"""
    
    def __init__(self, action_space_size: int = 4):
        super(BreakoutCNN, self).__init__()
        
        # Convolutional layers (same as DQN)
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Calculate the size of flattened features
        # For 84x84 input: (84-8)/4+1=20, (20-4)/2+1=9, (9-3)/1+1=7
        self.feature_size = 64 * 7 * 7
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.feature_size, 512)
        self.fc2 = nn.Linear(512, action_space_size)
        
    def forward(self, x):
        # Ensure input is in correct format: [batch, channels, height, width]
        if len(x.shape) == 4 and x.shape[-1] == 4:
            # Convert from [batch, height, width, channels] to [batch, channels, height, width]
            x = x.permute(0, 3, 1, 2).contiguous()
        elif len(x.shape) == 3 and x.shape[-1] == 4:
            # Convert from [height, width, channels] to [1, channels, height, width]
            x = x.permute(2, 0, 1).unsqueeze(0).contiguous()
        
        # Normalize input
        x = x.float() / 255.0
        
        # Convolutional layers with ReLU
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Flatten - use reshape instead of view for safety
        x = x.reshape(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        # Apply softmax for action probabilities
        return F.softmax(x, dim=-1)
    
    def get_action(self, state, deterministic=False):
        """Get action from policy"""
        with torch.no_grad():
            # Ensure proper tensor format
            if isinstance(state, np.ndarray):
                state = torch.tensor(state, dtype=torch.float32)
            
            if len(state.shape) == 3:
                # Add batch dimension: [H, W, C] -> [1, H, W, C]
                state = state.unsqueeze(0)
            
            action_probs = self.forward(state)
            
            if deterministic:
                action = torch.argmax(action_probs, dim=-1)
            else:
                action = torch.multinomial(action_probs, 1).squeeze(-1)
            
            return action.cpu().numpy()


class EvolutionaryStrategy:
    """Evolutionary Strategy optimization for neural networks"""
    
    def __init__(
        self,
        network: nn.Module,
        population_size: int = 50,
        sigma: float = 0.1,
        learning_rate: float = 0.01,
        decay: float = 0.999,
        antithetic: bool = True
    ):
        self.network = network
        self.population_size = population_size
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.decay = decay
        self.antithetic = antithetic
        
        # Get the parameter shapes
        self.param_shapes = []
        self.param_sizes = []
        for param in self.network.parameters():
            self.param_shapes.append(param.shape)
            self.param_sizes.append(param.numel())
        
        self.total_params = sum(self.param_sizes)
        
    def sample_population(self) -> List[np.ndarray]:
        """Sample a population of parameter perturbations"""
        if self.antithetic:
            # Use antithetic sampling for better variance reduction
            half_pop = self.population_size // 2
            epsilons = [np.random.randn(self.total_params) for _ in range(half_pop)]
            population = epsilons + [-eps for eps in epsilons]
        else:
            population = [np.random.randn(self.total_params) for _ in range(self.population_size)]
        
        return population
    
    def set_params(self, flat_params: np.ndarray):
        """Set network parameters from flat array"""
        start_idx = 0
        with torch.no_grad():
            for param, size, shape in zip(self.network.parameters(), self.param_sizes, self.param_shapes):
                param_slice = flat_params[start_idx:start_idx + size]
                param.copy_(torch.tensor(param_slice.reshape(shape), dtype=param.dtype))
                start_idx += size
    
    def get_params(self) -> np.ndarray:
        """Get network parameters as flat array"""
        params = []
        for param in self.network.parameters():
            params.append(param.detach().cpu().numpy().flatten())
        return np.concatenate(params)
    
    def update(self, fitness_scores: List[float], population: List[np.ndarray]):
        """Update network parameters based on fitness scores"""
        fitness_scores = np.array(fitness_scores)
        
        # Normalize fitness scores
        if np.std(fitness_scores) > 0:
            fitness_scores = (fitness_scores - np.mean(fitness_scores)) / np.std(fitness_scores)
        
        # Calculate update direction
        update = np.zeros(self.total_params)
        for i, epsilon in enumerate(population):
            update += fitness_scores[i] * epsilon
        
        update = self.learning_rate / (self.population_size * self.sigma) * update
        
        # Apply update
        current_params = self.get_params()
        new_params = current_params + update
        self.set_params(new_params)
        
        # Decay learning rate
        self.learning_rate *= self.decay


def evaluate_network(args):
    """Evaluate a network configuration"""
    network_params, eval_episodes, env_id, seed = args
    
    # Create environment
    env = make_atari_env(env_id, n_envs=1, seed=seed)
    env = VecFrameStack(env, n_stack=4)
    
    # Create network and set parameters
    network = BreakoutCNN(action_space_size=4)
    network.eval()
    
    # Set parameters if provided
    if network_params is not None:
        flat_params = network_params
        start_idx = 0
        with torch.no_grad():
            for param in network.parameters():
                param_size = param.numel()
                param_slice = flat_params[start_idx:start_idx + param_size]
                param.copy_(torch.tensor(param_slice.reshape(param.shape), dtype=param.dtype))
                start_idx += param_size
    
    total_reward = 0
    
    for _ in range(eval_episodes):
        obs = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # Convert observation to tensor - obs is already numpy array from VecFrameStack
            # VecFrameStack returns shape [1, 84, 84, 4] 
            obs_tensor = torch.tensor(obs, dtype=torch.float32)
            action = network.get_action(obs_tensor, deterministic=True)
            obs, reward, done, _ = env.step(action)
            episode_reward += reward[0]
        
        total_reward += episode_reward
    
    env.close()
    return total_reward / eval_episodes


class BreakoutES:
    """Breakout Evolution Strategy trainer"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Create environment for getting action space info
        self.env = make_atari_env("ALE/Breakout-v5", n_envs=1, seed=42)
        self.env = VecFrameStack(self.env, n_stack=4)
        
        # Create the neural network
        self.network = BreakoutCNN(action_space_size=4)
        
        # Create the ES optimizer
        self.es = EvolutionaryStrategy(
            network=self.network,
            population_size=config["population_size"],
            sigma=config["sigma"],
            learning_rate=config["learning_rate"],
            decay=config["decay"],
            antithetic=config["antithetic"]
        )
        
        # Best model tracking
        self.best_fitness = float('-inf')
        self.best_params = None
        
    def train(self, generations: int, eval_episodes: int = 5):
        """Train the network using Evolution Strategy"""
        
        for generation in range(generations):
            print(f"Generation {generation + 1}/{generations}")
            
            # Sample population
            population = self.es.sample_population()
            
            # Get current parameters
            base_params = self.es.get_params()
            
            # Create parameter sets for evaluation
            param_sets = []
            for epsilon in population:
                perturbed_params = base_params + self.config["sigma"] * epsilon
                param_sets.append(perturbed_params)
            
            # Evaluate population (can be parallelized)
            if self.config["parallel"]:
                # Parallel evaluation
                with Pool(processes=min(cpu_count(), len(param_sets))) as pool:
                    eval_args = [(params, eval_episodes, "ALE/Breakout-v5", 42 + i) 
                                for i, params in enumerate(param_sets)]
                    fitness_scores = pool.map(evaluate_network, eval_args)
            else:
                # Sequential evaluation
                fitness_scores = []
                for i, params in enumerate(param_sets):
                    fitness = evaluate_network((params, eval_episodes, "ALE/Breakout-v5", 42 + i))
                    fitness_scores.append(fitness)
            
            # Update network
            self.es.update(fitness_scores, population)
            
            # Track best model
            max_fitness = max(fitness_scores)
            avg_fitness = np.mean(fitness_scores)
            
            if max_fitness > self.best_fitness:
                self.best_fitness = max_fitness
                best_idx = fitness_scores.index(max_fitness)
                self.best_params = base_params + self.config["sigma"] * population[best_idx]
            
            # Log results
            print(f"Max fitness: {max_fitness:.2f}, Avg fitness: {avg_fitness:.2f}, Best ever: {self.best_fitness:.2f}")
            
            # Log to wandb if available
            if wandb.run is not None:
                wandb.log({
                    "generation": generation,
                    "max_fitness": max_fitness,
                    "avg_fitness": avg_fitness,
                    "best_fitness": self.best_fitness,
                    "learning_rate": self.es.learning_rate,
                })
            
            # Save model periodically
            if (generation + 1) % self.config["save_freq"] == 0:
                self.save_model(generation + 1)
    
    def save_model(self, generation: int):
        """Save the best model"""
        if self.best_params is not None:
            save_data = {
                'params': self.best_params,
                'fitness': self.best_fitness,
                'generation': generation,
                'config': self.config
            }
            
            filename = f"{self.config['models_dir']}/best_model_gen_{generation}.pkl"
            with open(filename, 'wb') as f:
                pickle.dump(save_data, f)
            
            print(f"Saved best model to {filename}")
    
    def load_model(self, filepath: str):
        """Load a saved model"""
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)
        
        self.best_params = save_data['params']
        self.best_fitness = save_data['fitness']
        self.es.set_params(self.best_params)
        
        print(f"Loaded model with fitness: {self.best_fitness}")
    
    def evaluate(self, episodes: int = 10, render: bool = False):
        """Evaluate the current best model"""
        if self.best_params is not None:
            self.es.set_params(self.best_params)
        
        total_reward = 0
        
        for episode in range(episodes):
            obs = self.env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                if render:
                    self.env.render()
                
                # obs is shape [1, 84, 84, 4] from VecFrameStack
                obs_tensor = torch.tensor(obs, dtype=torch.float32)
                action = self.network.get_action(obs_tensor, deterministic=True)
                obs, reward, done, _ = self.env.step(action)
                episode_reward += reward[0]
            
            total_reward += episode_reward
            print(f"Episode {episode + 1}: {episode_reward}")
        
        avg_reward = total_reward / episodes
        print(f"Average reward over {episodes} episodes: {avg_reward}")
        return avg_reward


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--parallel", action="store_true", default=True, help="Use parallel evaluation")
    parser.add_argument("--load_model", type=str, default=None, help="Path to load a saved model")
    parser.add_argument("--eval_only", action="store_true", default=False, help="Only evaluate, don't train")
    args = parser.parse_args()
    
    # Create directories
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    subdir = f"run_{current_datetime}"
    models_dir = os.path.join("ES_models", subdir)
    logdir = os.path.join("ES_logs", subdir)
    
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    
    # ES configuration
    es_config = {
        "population_size": 100,
        "sigma": 0.1,
        "learning_rate": 0.01,
        "decay": 0.995,
        "antithetic": True,
        "parallel": args.parallel,
        "models_dir": models_dir,
        "save_freq": 10,
    }
    
    training_config = {
        "generations": 200,
        "eval_episodes": 5,
    }
    
    combined_config = {
        **es_config,
        **training_config
    }
    
    # Initialize wandb
    if not args.eval_only:
        wandb.init(
            project="breakout-es",
            name=f"es_{current_datetime}",
            config=combined_config
        )
    
    # Create trainer
    trainer = BreakoutES(es_config)
    
    # Load model if specified
    if args.load_model:
        trainer.load_model(args.load_model)
    
    if args.eval_only:
        # Only evaluate
        trainer.evaluate(episodes=10, render=False)
    else:
        # Train the model
        trainer.train(
            generations=training_config["generations"],
            eval_episodes=training_config["eval_episodes"]
        )
        
        # Final evaluation
        print("\nFinal evaluation:")
        trainer.evaluate(episodes=10)
        
        wandb.finish()


if __name__ == "__main__":
    main()