import gymnasium as gym
import numpy as np
import time
from stable_baselines3 import PPO

from MeanRewardCallback import MeanRewardCallback
from MyReacherEnv import MyReacherEnv
import matplotlib.pyplot as plt
import pybullet as p

env=MyReacherEnv(gym.make("ErgoReacher-Headless-Simple-v1"))
env.reset()

mean_reward_callback=MeanRewardCallback(verbose=1)

model = PPO("MlpPolicy", env,verbose=1)

start_time=time.time()
model.learn(total_timesteps=81920, callback=mean_reward_callback)
print(f"Time: {time.time()-start_time}")

env.close()

graphical_env=MyReacherEnv(gym.make("ErgoReacher-Graphical-Simple-v1"))
obs=graphical_env.reset()[0]

final_rewards=[]

for i in range(1000):
    action, _states = model.predict(obs)
    time.sleep(0.2)
    obs, reward, terminated, truncated, info = graphical_env.step(action)
    final_rewards.append(reward)
    if terminated or truncated:
        break

plt.plot(final_rewards)
plt.show()
plt.close()
