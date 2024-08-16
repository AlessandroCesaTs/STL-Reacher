import gymnasium as gym
import pybullet as p
import time
from MyReacherEnv import MyReacherEnv

env=MyReacherEnv(gym.make("ErgoReacher-Graphical-Simple-v1"))

env.reset()

print(f" end effector is at {p.getLinkState(1,13)[0]}")

for i in range(50):
    time.sleep(0.1)
    obs, reward, terminated, truncated, info=env.step([1,-0.5,-1,1])
    print(f"distance is {info['distance']}")
    print(f"reward is {reward}")
for i in range(50):
    time.sleep(0.1)
    obs, reward, terminated, truncated, info=env.step([1,0,0.5,1])
    print(f"distance is {info['distance']}")
    print(f"reward is {reward}")

print(f" end effector is at {p.getLinkState(1,13)[0]}")
time.sleep(5)
env.close()

