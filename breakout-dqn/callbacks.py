import wandb
from stable_baselines3.common.callbacks import BaseCallback

# Custom callback for wandb logging
class WandbCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        
    def _on_step(self):
        # Log training info
        if len(self.model.ep_info_buffer) > 0:
            for info in self.model.ep_info_buffer:
                if 'r' in info:
                    wandb.log({"episode_reward": info['r']})
                if 'l' in info:
                    wandb.log({"episode_length": info['l']})
        return True