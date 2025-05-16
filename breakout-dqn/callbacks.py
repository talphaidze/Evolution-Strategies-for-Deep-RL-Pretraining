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
        if len(self.model.ep_info_buffer) > 0 and len(self.model.ep_info_buffer[-1]) > 0:
            info = self.model.ep_info_buffer[-1]
            metrics = {}
            if 'r' in info:
                metrics["episode_reward"] = info['r']
            if 'l' in info:
                metrics["episode_length"] = info['l']
            
            # Only log if we have metrics and use the current training step
            if metrics:
                wandb.log(metrics, step=self.num_timesteps)
        return True