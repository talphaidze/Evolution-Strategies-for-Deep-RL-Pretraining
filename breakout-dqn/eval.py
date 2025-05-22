from datetime import datetime
import os
import numpy as np
import argparse
import torch
import cv2
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.dqn import DQN

import ale_py

def load_latest_model(models_dir):
    # Get all subdirectories that match our run format
    subdirs = [f for f in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, f)) and f.startswith('run_')]
    if not subdirs:
        raise ValueError(f"No run directories found in {models_dir}")
    
    # Parse the datetime from each subdir (removing 'run_' prefix) and find the latest
    latest_subdir = max(subdirs, key=lambda x: datetime.strptime(x[4:], "%Y-%m-%d_%H-%M-%S"))
    
    # Look for model files in the latest subdir
    subdir_path = os.path.join(models_dir, latest_subdir)
    model_files = [f for f in os.listdir(subdir_path) if f.endswith('.zip')]
    if not model_files:
        raise ValueError(f"No model files found in {subdir_path}")
    
    # Get the latest model file based on the timestep number
    latest_model = max(model_files, key=lambda x: int(x.split('.')[0]))
    model_path = os.path.join(subdir_path, latest_model)
    
    return model_path, latest_model

def main():
    parser = argparse.ArgumentParser(description='Evaluate Breakout DQN model')
    parser.add_argument('--es', action='store_true', help='Use ES checkpoint')
    parser.add_argument('--dqn', action='store_true', help='Use DQN model')
    parser.add_argument('--model', type=str, help='Path to the model file to use (e.g., DQN_tmp/10000.zip or es_checkpoints/es_checkpoint_final.pt)')
    parser.add_argument('--episodes', type=int, default=5, help='Number of episodes to record')
    parser.add_argument('--save_video', action='store_true', help='Save gameplay videos')
    parser.add_argument('--video_dir', type=str, default='gameplay_videos', help='Directory to save videos')
    args = parser.parse_args()

    # Create video directory if saving videos
    if args.save_video:
        os.makedirs(args.video_dir, exist_ok=True)

    # Create the environment
    env = make_atari_env(
        "ALE/Breakout-v5",
        n_envs=1,
        seed=42,
    )
    env = VecFrameStack(env, n_stack=4)

    # Load the model
    if args.es and args.dqn:
        raise ValueError("Cannot specify both --es and --dqn")
    if args.es:
        models_dir = "DQN_es_models"
        model_path, model_name = load_latest_model(models_dir)
        print(f"Using latest ES checkpoint: {model_path}")
    elif args.dqn:
        models_dir = "DQN_sb3_models"
        model_path, model_name = load_latest_model(models_dir)
        print(f"Using latest DQN model: {model_path}")
    elif args.model:
        model_path = args.model
        model_name = os.path.basename(model_path)
        print(f"Using specified model: {model_path}")
    else:
        print("No model specified")
        models_dir = "DQN_sb3_models"
        model_path, model_name = load_latest_model(models_dir)
        print(f"Using latest DQN model: {model_path}")
    
    # Check if it's an ES checkpoint or DQN model
    if model_path.endswith('.pt'):
        # Load ES checkpoint
        checkpoint = torch.load(model_path)
        model = DQN(
            "CnnPolicy",
            env,
            learning_rate=2e-4,
            buffer_size=50_000,
            batch_size=32,
            learning_starts=100_000,
            target_update_interval=1000,
            train_freq=4,
            gradient_steps=1,
            exploration_fraction=0.1,
            exploration_final_eps=0.01,
            optimize_memory_usage=False,
            verbose=0,
        )
        model.policy.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded ES checkpoint from generation {checkpoint['generation']}")
        print(f"Checkpoint metrics: {checkpoint['metrics']}")
    else:
        # Load DQN model
        model = DQN.load(model_path)

    print(f"\nStarting {args.episodes} episodes of gameplay...")
    cv2.namedWindow('Breakout', cv2.WINDOW_NORMAL)
    
    total_rewards = []
    total_episode_lengths = []
    
    # Actions: 0=NOOP, 1=FIRE, 2=RIGHT, 3=LEFT
    FIRE_ACTION = 1
    
    for episode in range(args.episodes):
        print(f"\nEpisode {episode + 1}/{args.episodes}")
        obs = env.reset()
        done = False
        episode_reward = 0
        lives = 5
        episode_length = 0
        
        # Initialize video writer if saving videos
        if args.save_video:
            video_path = os.path.join(args.video_dir, f'episode_{episode+1}.mp4')
            # Get frame dimensions from first frame
            frame = env.envs[0].env.env.render()
            if isinstance(frame, np.ndarray):
                if len(frame.shape) == 2:  # If grayscale, convert to RGB
                    frame = np.stack([frame] * 3, axis=-1)
                if frame.shape[0] < frame.shape[1]:  # If frame is transposed
                    frame = frame.transpose(1, 0, 2)
                height, width = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(video_path, fourcc, 30.0, (width, height))
        
        # Start the game by firing the ball
        obs, rewards, dones, _ = env.step([FIRE_ACTION])
        episode_length += 1
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, info = env.step(action)
            episode_length += 1
            episode_reward += rewards[0]
            done = dones[0]
            # Get and display the current frame
            frame = env.envs[0].env.env.render()
            if isinstance(frame, np.ndarray):
                if len(frame.shape) == 2:  # If grayscale, convert to RGB
                    frame = np.stack([frame] * 3, axis=-1)
                if frame.shape[0] < frame.shape[1]:  # If frame is transposed
                    frame = frame.transpose(1, 0, 2)
                
                # Display the frame
                cv2.imshow('Breakout', frame[:, :, ::-1])  # Convert RGB to BGR for OpenCV
                
                # Save frame if recording
                if args.save_video:
                    video_writer.write(frame[:, :, ::-1])  # Convert RGB to BGR for OpenCV
                
                if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
                    break

            # Fire to start new life if ball is lost
            if 'lives' in info[0] and info[0]['lives'] < lives:
                lives = info[0]['lives']
                obs, rewards, dones, _ = env.step([FIRE_ACTION])
                episode_length += 1
                # Get and display the current frame
                frame = env.envs[0].env.env.render()
                if isinstance(frame, np.ndarray):
                    if len(frame.shape) == 2:  # If grayscale, convert to RGB
                        frame = np.stack([frame] * 3, axis=-1)
                    if frame.shape[0] < frame.shape[1]:  # If frame is transposed
                        frame = frame.transpose(1, 0, 2)
                    
                    # Display the frame
                    cv2.imshow('Breakout', frame[:, :, ::-1])  # Convert RGB to BGR for OpenCV
                    
                    # Save frame if recording
                    if args.save_video:
                        video_writer.write(frame[:, :, ::-1])  # Convert RGB to BGR for OpenCV
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
                        break
            if done:
                print(f"Episode {episode + 1} finished with reward: {episode_reward:.2f}")
                cv2.waitKey(1000)  # Wait for 1 second between episodes
                break
        
        # Release video writer if saving videos
        if args.save_video:
            video_writer.release()
            print(f"Saved video to {video_path}")
        
        total_rewards.append(episode_reward)
        total_episode_lengths.append(episode_length)
        mean_reward = np.mean(total_rewards)
        mean_episode_length = np.mean(total_episode_lengths)

    print(f"\nFinal Results:")
    print(f"Average reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    print(f"Average steps: {np.mean(total_episode_lengths):.2f} ± {np.std(total_episode_lengths):.2f}")

    # Close everything
    cv2.destroyAllWindows()
    env.close()

if __name__ == "__main__":
    main()