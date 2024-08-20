import os
from Trainer import Trainer
from MyReacherEnv import MyReacherEnv
from stable_baselines3 import PPO

model=PPO.load('models/model.zip')

output_path=os.getcwd()

os.makedirs(os.path.join(output_path,'videos'), exist_ok=True)

video_path=os.path.join(output_path,'videos')

environment=MyReacherEnv(video_path)

model.set_env(environment)

trainer=Trainer(environment,model,output_path)

trainer.test()
