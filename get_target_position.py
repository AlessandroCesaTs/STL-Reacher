import gym
import pybullet as p
from MyEnv import MyEnv

env=MyEnv(gym.make("ErgoGripper-Square-Touch-Double-Graphical-v1",headless=True))

env.reset()

print(f" end effector is at {p.getLinkState(1,15)[0]}")

for i in range(15):
    env.step([0,2,0,2,0,2])  

print(f" end effector is at {p.getLinkState(1,15)[0]}")
