import os
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from classes.trainer import Trainer
from classes.single_reacher_env import SingleReacherEnv
from classes.double_reacher_env import DoubleReacherEnv
from utils.utils import get_num_cpus

if __name__=="__main__":
    
    parser=argparse.ArgumentParser()
    parser.add_argument('--output_path',type=str,default=os.path.join(os.getcwd(),'output'))
    parser.add_argument('--double',action=argparse.BooleanOptionalAction,default=False)
    parser.add_argument('--hard_reward',action=argparse.BooleanOptionalAction,default=False)
    parser.add_argument('--max_steps',type=int,default=10)
    parser.add_argument('--test_runs',type=int,default=1)
    parser.add_argument('--num_of_goals',type=int,default=3)

    args=parser.parse_args()
    double=args.double
    hard_reward=args.hard_reward
    output_path=args.output_path
    max_steps=args.max_steps
    test_runs=args.test_runs
    num_of_goals=args.num_of_goals
    n_envs=get_num_cpus()
   

    model_path=os.path.join(output_path,'model')

    if double:
        environment=make_vec_env(DoubleReacherEnv,n_envs=n_envs,vec_env_cls=SubprocVecEnv,env_kwargs={'max_steps':max_steps,'output_path':output_path})
    else:
        environment=make_vec_env(SingleReacherEnv,n_envs=n_envs,vec_env_cls=SubprocVecEnv,env_kwargs={'max_steps':max_steps,'output_path':output_path,'hard_reward':hard_reward})
        
    model=PPO.load(model_path,environment)

    trainer=Trainer(environment,model,output_path)

    if double:
        trainer.test_double(test_runs=test_runs)
    else:
        trainer.test_single(test_runs=test_runs)
    
    environment.close()
