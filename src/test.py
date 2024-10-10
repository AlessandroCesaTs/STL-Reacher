import os
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from classes.trainer import Trainer
from classes.my_reacher_env import MyReacherEnv
from utils.utils import get_num_cpus

if __name__=="__main__":
    
    parser=argparse.ArgumentParser()
    parser.add_argument('--output_path',type=str,default=os.path.join(os.getcwd(),'output'))
    parser.add_argument('--change_target',action=argparse.BooleanOptionalAction,default=False)
    parser.add_argument('--hard_reward',action=argparse.BooleanOptionalAction,default=False)
    parser.add_argument('--max_steps',type=int,default=10)
    parser.add_argument('--test_runs',type=int,default=1)
    parser.add_argument('--num_of_goals',type=int,default=3)

    args=parser.parse_args()
    change_target=args.change_target
    hard_reward=args.hard_reward
    output_path=args.output_path
    max_steps=args.max_steps
    test_runs=args.test_runs
    num_of_goals=args.num_of_goals
    n_envs=get_num_cpus()

    model_path=os.path.join(output_path,'model')

    environment=make_vec_env(MyReacherEnv,n_envs=n_envs,vec_env_cls=SubprocVecEnv,env_kwargs={'max_steps':max_steps,'output_path':output_path,'change_target':change_target,'hard_reward':hard_reward})
        
    model=PPO.load(model_path,environment)

    trainer=Trainer(environment,model,output_path)

    if change_target:
        trainer.test_changing_target(test_runs=test_runs)
    else:
        trainer.test_single_target(test_runs=test_runs)
    
    environment.close()
