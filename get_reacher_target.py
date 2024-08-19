import gymnasium as gym
import pybullet as p
import time
from MyReacherEnv import MyReacherEnv

env=MyReacherEnv(gym.make("ErgoReacher-Graphical-Simple-v1"))

env.reset()

print(f" end effector is at {p.getLinkState(1,13)[0]}")

for i in range(50):
    time.sleep(0.1)
    if i%2==0:
        obs, reward, terminated, truncated, info=env.step([0.26954633, -0.82377454, 0.18293788, -0.56170284, 0.10516296,
       0.79996014])
    else:
        obs, reward, terminated, truncated, info=env.step([0.41197872, -0.93032861, 0.19448059, 0.86295398, -0.963815  ,
       0.53447552])
    if i==15:
        print(f" end effector is at {p.getLinkState(1,13)[0]}")

print(f" end effector is at {p.getLinkState(1,13)[0]}")
time.sleep(5)
env.close()

