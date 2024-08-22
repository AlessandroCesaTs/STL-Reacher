import os
from classes.trainer import Trainer
from classes.my_reacher_env import MyReacherEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

if __name__=='__main__':
    

    output_path=os.getcwd()

    os.makedirs(os.path.join(output_path,'videos'), exist_ok=True)

    video_path=os.path.join(output_path,'videos')

    environment=make_vec_env(MyReacherEnv,n_envs=1,vec_env_cls=SubprocVecEnv,env_kwargs={'num_of_goals':3,'num_of_avoids':1,'video_path':video_path})
    #environment=MyReacherEnv(video_path=video_path)

    model=PPO.load('models/model.zip',environment)

    trainer=Trainer(environment,model,output_path)

    trainer.test(test_steps=10)
