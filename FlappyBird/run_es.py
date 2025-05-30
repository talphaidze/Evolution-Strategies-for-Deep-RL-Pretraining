import time
import torch
import numpy as np
from flappy_gym_env import FlappyBirdEnv
from train_es import MLP
from torch.nn.utils import vector_to_parameters


def main():
    # Initialize environment
    env = FlappyBirdEnv(display_screen=True)
    obs = env.reset()
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gen = 1
    # Build model and load pretrained weights
    model = MLP(obs_dim, action_dim, hidden_sizes=[64, 64]).to(device).double()
    flat = torch.load(f"best_policy_so_far_gen{gen}.pth", map_location=device)
    vector_to_parameters(flat, model.parameters())
    model.eval()


    total_reward = 0.0
    done = False
    obs = env.reset()

    try:
        while True:
            obs_t = torch.tensor(obs, dtype=torch.float64, device=device).unsqueeze(0).float()

            with torch.no_grad():
                probs = model(obs_t)
                action = torch.argmax(probs)

            obs, reward, done, _ = env.step(action)
            total_reward += reward
            time.sleep(1 / 30)

            if done:
                print(f"Episode done | Total reward: {total_reward:.2f}")
                obs = env.reset()
                total_reward = 0.0

    except KeyboardInterrupt:
        print("\nExiting viewer.")
    finally:
        try:
            env.close()
        except Exception:
            pass

if __name__ == "__main__":
    main()
