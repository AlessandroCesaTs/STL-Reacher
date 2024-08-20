import os
from stable_baselines3 import PPO
from Trainer import Trainer
from MyReacherEnv import MyReacherEnv

output_path=os.getcwd()

os.makedirs(os.path.join(output_path,'videos'), exist_ok=True)

video_path=os.path.join(output_path,'videos')

environment=MyReacherEnv(video_path)
model = PPO("MlpPolicy", environment)


trainer=Trainer(environment,model,output_path)

trainer.train()
trainer.test()
