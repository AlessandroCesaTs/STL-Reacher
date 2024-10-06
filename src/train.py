import os
import time
import argparse
import csv
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from classes.trainer import Trainer
from classes.my_reacher_env import MyReacherEnv
from utils.utils import get_num_cpus

urdf_dir='env/lib/python3.12/site-packages/gym_ergojr/scenes/'

if __name__=="__main__":
    start_time=time.time()

    parser=argparse.ArgumentParser()
    parser.add_argument('--output_path',type=str,default=os.path.join(os.getcwd(),'output'))
    parser.add_argument('--change_target',action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument('--total_timesteps',type=int,default=256)
    parser.add_argument('--n_steps',type=int,default=128)
    parser.add_argument('--max_steps',type=int,default=100)
    parser.add_argument('--n_epochs',type=int,default=2)

    args=parser.parse_args()
    output_path=args.output_path
    change_target=args.change_target
    total_timesteps=args.total_timesteps
    n_steps=args.n_steps
    max_steps=args.max_steps
    n_epochs=args.n_epochs
    #n_envs=get_num_cpus()
    n_envs=1

    os.makedirs(output_path,exist_ok=True)

    times_csv_path=os.path.join(output_path,'times.csv')

    environment=make_vec_env(MyReacherEnv,n_envs=n_envs,vec_env_cls=SubprocVecEnv,env_kwargs={'max_steps':max_steps,'output_path':output_path,'change_target':change_target})
    model = PPO("MlpPolicy", environment,n_steps=n_steps,n_epochs=n_epochs)

    trainer=Trainer(environment,model,output_path)
    
    trainer.train(total_timesteps=total_timesteps)
    
    environment.close()

    print(f"Total Time: {(time.time()-start_time)/60} minutes")
