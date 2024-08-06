import gymnasium as gym
import time
from stable_baselines3 import PPO
from MyEnv import MyEnv
import matplotlib.pyplot as plt

env=MyEnv(gym.make("ErgoGripper-Square-Touch-Double-Graphical-v1",headless=True))

env.reset()

model = PPO("MlpPolicy", env, verbose=1)
start_time=time.time()
model.learn(total_timesteps=250)
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

plt.plot(final_rewards)
plt.show()
plt.close()
