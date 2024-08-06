import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gym_ergojr.envs.ergo_gripper_env import ErgoGripperEnv
import itertools
import pybullet as p

class MyEnv(ErgoGripperEnv):
    def __init__(self, env,target=np.array((0.03167205854155157, 0.209124010039151, 0.18276742932051596))):
        super().__init__(env)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32)
        self.target=target

        self.actions = list(itertools.product([-1, 0, 1], repeat=6))
        self.action_space = spaces.MultiDiscrete([3,3,3,3,3,3])
    
    def reset(self, **kwargs):
        super().reset()
        observation=self.robot.observe()[:6]
        reset_info={} #needed for stable baseline
        return observation,reset_info
    
    def observation(self):
        return self.robot.observe()[:6]

    def move(self, action):
        observation, reward, done, info = super().step(action)
        reward=-np.linalg.norm(self.get_end_effector_position()-self.target)
        terminated=done
        truncated=done
        if reward>-0.01:
            terminated=True  #needing terminated and truncated for stable baseline
        return self.observation(), reward, terminated,truncated, info
    
    def step(self,action):
        action=self.map_action(action)
        target=self.robot.observe()[:6]
        for i in range(5):
            target[i]+=action[i]*0.1 
        return self.move(target)
    
    def get_end_effector_position(self):
        return np.array(p.getLinkState(1,15)[0])

    def map_action(self,action):
        return [x - 1 for x in action]  # Convert 0->-1, 1->0, 2->1

