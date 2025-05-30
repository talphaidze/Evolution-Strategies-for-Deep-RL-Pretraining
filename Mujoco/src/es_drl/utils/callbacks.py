# src/es_drl/utils/callbacks.py
"""
Training callbacks for monitoring and logging.
This module provides callback classes for tracking episode rewards and other
metrics during training, with integration to the Logger utility.
"""

from stable_baselines3.common.callbacks import BaseCallback


class EpisodeLoggerCallback(BaseCallback):
    """
    Logs episode rewards to a CSV via the Logger utility.
    Expects infos dict from Monitor wrapper with 'episode' key.
    """

    def __init__(self, data_logger, verbose=0):
        super().__init__(verbose)
        self.data_logger = data_logger
        self.episode_num = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            ep = info.get("episode")
            if ep is not None:
                # ep["r"] is the episode reward
                self.data_logger.log(self.episode_num, {"episode_reward": ep["r"]})
                self.episode_num += 1
        return True
