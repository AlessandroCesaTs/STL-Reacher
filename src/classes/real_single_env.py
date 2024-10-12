import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
import pickle


import numpy as np
class RealSingleenv(gym.Env):
    def __init(self):
        super().__init__()
        self.observation_space = spaces.Box(low=-1, high=1, shape=(15,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32)

        self.model=PPO.load('outputs/fixed_hard_const_3/model.zip')
        with open('outputs/fixed_hard_const_3/setting.pkl', 'rb') as f:
            setting = pickle.load(f)
        self.goal=setting['goal']
        self.initial_pose=setting['initial_pose']
        self.avoid=setting['avoid']

    def get_action(self,obs):
    