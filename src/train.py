import os
import time
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from classes.trainer import Trainer
from classes.single_reacher_env import SingleReacherEnv
from classes.double_reacher_env import DoubleReacherEnv
from utils.utils import get_num_cpus

urdf_dir='env/lib/python3.12/site-packages/gym_ergojr/scenes/'

if __name__=="__main__":
    start_time=time.time()

    parser=argparse.ArgumentParser()
    parser.add_argument('--output_path',type=str,default=os.path.join(os.getcwd(),'output'))
    parser.add_argument('--double',action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument('--hard_reward',action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument('--total_timesteps',type=int,default=256)
    parser.add_argument('--n_steps',type=int,default=128)
    parser.add_argument('--max_steps',type=int,default=100)
    parser.add_argument('--n_epochs',type=int,default=2)

    args=parser.parse_args()
    output_path=args.output_path
    double=args.double
    hard_reward=args.hard_reward
    total_timesteps=args.total_timesteps
    n_steps=args.n_steps
    max_steps=args.max_steps
    n_epochs=args.n_epochs
    n_envs=get_num_cpus()

    os.makedirs(output_path,exist_ok=True)

    if double:
        environment=make_vec_env(DoubleReacherEnv,n_envs=n_envs,vec_env_cls=SubprocVecEnv,env_kwargs={'max_steps':max_steps,'output_path':output_path})
    else:
        environment=make_vec_env(SingleReacherEnv,n_envs=n_envs,vec_env_cls=SubprocVecEnv,env_kwargs={'max_steps':max_steps,'output_path':output_path,'hard_reward':hard_reward})
    model = PPO("MlpPolicy", environment,n_steps=n_steps,n_epochs=n_epochs)

    trainer=Trainer(environment,model,output_path)
    
    trainer.train(total_timesteps=total_timesteps)

    if not double:
        environment.env_method("save_setting_to_file",os.path.join(output_path,'setting.pkl'),indices=0)
    
    environment.close()

    print(f"Total Time: {(time.time()-start_time)/60} minutes")
