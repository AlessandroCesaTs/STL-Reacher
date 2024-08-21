import os
from stable_baselines3 import PPO
from Trainer import Trainer
from MyReacherEnv import MyReacherEnv

output_path=os.getcwd()

os.makedirs(os.path.join(output_path,'videos'), exist_ok=True)

video_path=os.path.join(output_path,'videos')

environment=MyReacherEnv(num_of_goals=3,num_of_avoids=1,video_path=video_path)
model = PPO("MlpPolicy", environment)

trainer=Trainer(environment,model,output_path)
trainer.train(total_timesteps=4096)
print("Testing")
trainer.test()
