import os
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from classes.trainer import Trainer
from classes.my_reacher_env import MyReacherEnv

if __name__=="__main__":

    parser=argparse.ArgumentParser()
    parser.add_argument('--output_path',type=str,default=os.getcwd())
    parser.add_argument('--total_timesteps',type=int,default=2048)
    parser.add_argument('--n_envs',type=int,default=1)
    parser.add_argument('--num_of_goals',type=int,default=1)
    parser.add_argument('--num_of_avoids',type=int,default=1)

    args=parser.parse_args()
    output_path=args.output_path
    total_timesteps=args.total_timesteps
    n_envs=args.n_envs
    num_of_goals=args.num_of_goals
    num_of_avoids=args.num_of_avoids

    environment=make_vec_env(MyReacherEnv,n_envs=n_envs,vec_env_cls=SubprocVecEnv,env_kwargs={'num_of_goals':num_of_goals,'num_of_avoids':num_of_avoids,'output_path':output_path})
        
    model = PPO("MlpPolicy", environment)

    trainer=Trainer(environment,model,output_path)
    
    trainer.train(total_timesteps=total_timesteps)
    
    environment.close()
