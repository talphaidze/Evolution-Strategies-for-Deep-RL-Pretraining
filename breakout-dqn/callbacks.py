import wandb
from stable_baselines3.common.callbacks import BaseCallback
import time
# Custom callback for wandb logging
class WandbCallback(BaseCallback):
    def __init__(self, start_time, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.start_time = start_time
        
    def _on_step(self):
        if len(self.model.ep_info_buffer) > 0:
            for info in self.model.ep_info_buffer:
                log_data = {}
                log_data["time_taken(min)"] = (time.time() - self.start_time)/60
                if 'r' in info:
                    log_data["episode_reward"] = info['r']
                if 'l' in info:
                    log_data["episode_length"] = info['l']
                if log_data:
                    wandb.log(log_data, step=self.num_timesteps)
        return True
