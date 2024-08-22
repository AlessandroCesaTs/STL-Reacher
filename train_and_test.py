import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from Trainer import Trainer
from MyReacherEnv import MyReacherEnv

if __name__=="__main__":
    output_path=os.getcwd()

    os.makedirs(os.path.join(output_path,'videos'), exist_ok=True)

    video_path=os.path.join(output_path,'videos')

    environment=make_vec_env(MyReacherEnv,n_envs=2,vec_env_cls=SubprocVecEnv,env_kwargs={'num_of_goals':3,'num_of_avoids':1,'video_path':video_path})

        
    model = PPO("MlpPolicy", environment)

    trainer=Trainer(environment,model,output_path)
    trainer.train(total_timesteps=4096)
    print("Testing")
    trainer.test()
    

    environment.close()
