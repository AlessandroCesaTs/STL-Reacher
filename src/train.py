import os
import time
import argparse
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from classes.trainer import Trainer
from classes.my_reacher_env import MyReacherEnv
from utils.utils import get_num_cpus

def make_env(rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = MyReacherEnv(num_of_goals=3,num_of_avoids=1,output_path=os.getcwd())
        #env.seed(seed + rank)
        return env
    set_random_seed(seed)
    return _init


if __name__=="__main__":
    start_time=time.time()

    parser=argparse.ArgumentParser()
    parser.add_argument('--output_path',type=str,default=os.getcwd())
    parser.add_argument('--total_timesteps',type=int,default=4096)
    parser.add_argument('--num_of_goals',type=int,default=3)
    parser.add_argument('--num_of_avoids',type=int,default=1)

    args=parser.parse_args()
    output_path=args.output_path
    total_timesteps=args.total_timesteps
    num_of_goals=args.num_of_goals
    num_of_avoids=args.num_of_avoids
    n_envs=get_num_cpus()
    #n_envs=1

    #environment=make_vec_env(MyReacherEnv,n_envs=n_envs,vec_env_cls=SubprocVecEnv,env_kwargs={'num_of_goals':num_of_goals,'num_of_avoids':num_of_avoids,'output_path':output_path})
    environment=SubprocVecEnv([make_env(i) for i in range(n_envs)])
    model = PPO("MlpPolicy", environment)

    trainer=Trainer(environment,model,output_path)
    
    trainer.train(total_timesteps=total_timesteps)
    
    environment.close()

    print(f"Total Time: {(time.time()-start_time)/60} minutes")
