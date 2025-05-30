import time
import numpy as np
import cv2
import pygame
from stable_baselines3 import DQN
from flappy_gym_env import FlappyBirdEnv

def record_pygame_screen(filename, surface, fps=30):
    width, height = surface.get_size()
    video = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    return video

def pygame_surface_to_cvimage(surface):
    rgb_array = pygame.surfarray.array3d(surface)
    return cv2.cvtColor(np.transpose(rgb_array, (1, 0, 2)), cv2.COLOR_RGB2BGR)

def main():
    env = FlappyBirdEnv(display_screen=True)
    model = DQN.load("dqn_flappy", env=env)

    total_reward = 0
    obs = env.reset()
    done = False
    episode = 0

    video_writer = None
    video = False

    try:
        while True:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            total_reward += reward

            if video:
                surface = pygame.display.get_surface()
                if surface:
                    if video_writer is None:
                        video_writer = record_pygame_screen(f"flappy_ep{episode+1}.mp4", surface)
                    frame = pygame_surface_to_cvimage(surface)
                    video_writer.write(frame)

            time.sleep(1 / 30)

            if done:
                print(f"Episode {episode + 1} done | Total reward: {total_reward:.2f}")
                obs = env.reset()
                total_reward = 0
                if video_writer:
                    video_writer.release()
                    video_writer = None
                episode += 1

    except KeyboardInterrupt:
        print("\n👋 Exiting viewer.")
        if video_writer:
            video_writer.release()
    finally:
        env.close()
        if video_writer:
            video_writer.release()

if __name__ == "__main__":
    main()
