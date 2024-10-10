import os
import numpy as np
from stable_baselines3.common import base_class
from stable_baselines3.common.callbacks import BaseCallback

from utils.utils import write_to_csv

class MyCallback(BaseCallback):
    def __init__(self,logs_path):
        super().__init__()
        self.logs_path=logs_path

        self.rewards_log_path=os.path.join(self.logs_path,'training.csv')
        self.final_state_log_path=os.path.join(self.logs_path,'end_conditions.csv')

        write_to_csv(self.rewards_log_path,['Environment','Episode','Step','Robustness','Reward'],'w')
        write_to_csv(self.final_state_log_path,['Environment','Episode','Robustness','Reward','End_Condition'],'w')        

    def init_callback(self, model: "base_class.BaseAlgorithm") -> None:
        """
        Initialize the callback by saving references to the
        RL model and the training environment for convenience.
        """
        super().init_callback(model)
        self.num_envs=self.training_env.num_envs

    def _on_step(self):

        infos=self.locals['infos']
        rewards=self.locals['rewards']
        dones=self.locals['dones']

        for env_index in range(self.num_envs):
            info=infos[env_index]
            episode=info['episode_number']
            step=info['step']
            robustness=info['requirement_robustness']
            reward=rewards[env_index]

            write_to_csv(self.rewards_log_path,[env_index,episode,step,robustness,reward],'a')
                    
            if dones[env_index]:
                end_condition=info['end_condition']
                write_to_csv(self.final_state_log_path,[env_index,episode,robustness,reward,end_condition],'a')

        return True
    
