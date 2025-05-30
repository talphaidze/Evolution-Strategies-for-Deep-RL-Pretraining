import gym
import random
import numpy as np
from ple import PLE
from ple.games.flappybird import FlappyBird
from gym import spaces
from gym.utils import seeding

class FlappyBirdEnv(gym.Env):
    """
    FlappyBirdEnv with a proper `seed=` param that PLE will honor.
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self, display_screen=False, frame_skip=4, seed=0):
        super().__init__()
        self.frame_skip = frame_skip
        self.np_random, seed = seeding.np_random(seed)
        random.seed(seed) 

        self.game = FlappyBird()
        self.p = PLE(
            self.game,
            fps=30,
            display_screen=display_screen,
            rng=int(seed),
        )
        self.p.init()

        self.action_set   = self.p.getActionSet()
        self.action_space = spaces.Discrete(len(self.action_set))
        self.action_space.seed(seed)

        example_state = self.p.getGameState()
        self.observation_space = spaces.Box(
            -np.inf, np.inf,
            shape=(len(example_state),),
            dtype=np.float32,
        )
        self.observation_space.seed(seed)

    def reset(self):
        self.p.reset_game()
        return self._get_obs()

    def step(self, action):
        total_reward = 0.0
        done = False
        for _ in range(self.frame_skip):
            reward = self.p.act(self.action_set[action])
            total_reward += reward + 0.1
            done = self.p.game_over()
            if done:
                break
        return self._get_obs(), total_reward, done, {}

    def _get_obs(self):
        state = self.p.getGameState()
        return np.array(list(state.values()), dtype=np.float32)

    def render(self, mode="human"):
        pass

    def close(self):
        self.p.quit()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        random.seed(seed)
        return [seed]
