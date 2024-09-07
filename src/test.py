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
    parser.add_argument('--output_path',type=str,default=os.getcwd())
    parser.add_argument('--model_path',type=str,default=os.path.join(os.getcwd(),'models','model.zip'))
    parser.add_argument('--test_steps',type=int,default=100)
    parser.add_argument('--num_of_goals',type=int,default=3)
    parser.add_argument('--num_of_avoids',type=int,default=1)

    args=parser.parse_args()
    output_path=args.output_path
    model_path=args.model_path
    test_steps=args.test_steps
    num_of_goals=args.num_of_goals
    num_of_avoids=args.num_of_avoids
    n_envs=get_num_cpus()

    environment=make_vec_env(MyReacherEnv,n_envs=n_envs,vec_env_cls=SubprocVecEnv,env_kwargs={'num_of_goals':num_of_goals,'num_of_avoids':num_of_avoids,'output_path':output_path})
        
    model=PPO.load(model_path,environment)

    trainer=Trainer(environment,model,output_path)

    trainer.test(max_test_steps=test_steps)
    
    environment.close()
