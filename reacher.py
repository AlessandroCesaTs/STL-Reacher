import gymnasium as gym
import time
from stable_baselines3 import PPO
from MyReacherEnv import MyReacherEnv
import matplotlib.pyplot as plt
import pybullet as p

env=MyReacherEnv(gym.make("ErgoReacher-Headless-Simple-v1"))
env.reset()


model = PPO("MlpPolicy", env, verbose=1)
start_time=time.time()
model.learn(total_timesteps=250000)
print(f"Time: {time.time()-start_time}")
buffer=model.rollout_buffer
rewards=buffer.rewards
plt.plot(rewards)
plt.show()
plt.close()

obs=env.reset()[0]

final_rewards=[]

for i in range(100):
    action, _states = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    final_rewards.append(reward)
    if terminated or truncated:
        break

plt.plot(final_rewards)
plt.show()
plt.close()
