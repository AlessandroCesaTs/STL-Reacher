import os
import numpy as np
from stable_baselines3.common import base_class
from stable_baselines3.common.callbacks import BaseCallback

from utils.utils import write_to_csv

class MyCallback(BaseCallback):
    def __init__(self,logs_path):
        super().__init__()
        self.logs_path=logs_path

        self.rewards_log_path=os.path.join(self.logs_path,'rewards.csv')
        self.final_robustness_log_path=os.path.join(self.logs_path,'final_robustness.csv')
        self.final_boolean_log_path=os.path.join(self.logs_path,'final_boolean.csv')

        write_to_csv(self.rewards_log_path,['Environment','Episode','Step','Reward'],'w')
        write_to_csv(self.final_robustness_log_path,['Environment','Episode','Robustness'],'w')
        write_to_csv(self.final_boolean_log_path,['Environment','Episode','Boolean'],'w')
        

    def init_callback(self, model: "base_class.BaseAlgorithm") -> None:
        """
        Initialize the callback by saving references to the
        RL model and the training environment for convenience.
        """
        super().init_callback(model)
        self.num_envs=self.training_env.num_envs
        self.number_of_formulas_to_monitor=self.training_env.get_attr('number_of_formulas_to_monitor',indices=0)[0]

        self.robustnesses_log_paths=[os.path.join(self.logs_path,f"robustness_{i}.csv") for i in range(self.number_of_formulas_to_monitor)]
        for robustness_index in range(self.number_of_formulas_to_monitor):
            write_to_csv(self.robustnesses_log_paths[robustness_index],['Environment','Episode','Step','Robustness'],'w')

    def _on_step(self):

        infos=self.locals['infos']
        rewards=self.locals['rewards']
        dones=self.locals['dones']

        for env_index in range(self.num_envs):
            info=infos[env_index]
            episode=info['episode_number']
            step=info['step']
            goal_to_reach=info['goal_to_reach']
            reward=rewards[env_index]

            write_to_csv(self.rewards_log_path,[env_index,episode,step,reward],'a')

            write_to_csv(self.robustnesses_log_paths[goal_to_reach],[env_index,episode,step,reward],'a')
                    
            if dones[env_index]:
                final_robustness=info['final_robustness']
                final_boolean=info['final_boolean']
                write_to_csv(self.final_robustness_log_path,[env_index,episode,final_robustness],'a')
                write_to_csv(self.final_boolean_log_path,[env_index,episode,final_boolean],'a')

        return True
    
