import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common import base_class
from stable_baselines3.common.callbacks import BaseCallback
class MyCallback(BaseCallback):

    def __init__(self,rewards_path,plots_path):
        super().__init__()
        self.rewards_path=rewards_path
        self.rewards_plot_path=os.path.join(plots_path,'rewards.png')
        self.robustnesses_plot_paths=[os.path.join(plots_path,'robustness_'+str(i)+'.png') for i in range(3)]
        self.bools_plot_path=os.path.join(plots_path,'bools.png')
        self.mean_rewards=[]
        self.mean_robustnesses=[[]for _ in range(3)]
        self.bool_terminations=[]
        self.tot_episodes=0
        with open(self.rewards_path,mode='w',newline='') as file:
            writer=csv.writer(file)
            writer.writerow(['Episode','Mean Reward'])

    def init_callback(self, model: "base_class.BaseAlgorithm") -> None:
        """
        Initialize the callback by saving references to the
        RL model and the training environment for convenience.
        """
        super().init_callback(model)
        self.num_envs=self.training_env.num_envs
        self.current_rewards=np.zeros(self.num_envs)
        self.current_robustnesses=np.zeros((self.num_envs,3))
        self.current_lengths = np.zeros(self.num_envs)
        self.episodes=np.zeros(self.num_envs)

    def _on_step(self):
        self.current_rewards+=self.locals['rewards']
        self.current_lengths += 1
        for env_index in range(self.num_envs):
            self.current_robustnesses[env_index]+=self.locals['infos'][env_index]['robustnesses']                        
            if self.locals['dones'][env_index]:
                mean_reward = self.current_rewards[env_index]/self.current_lengths[env_index]
                mean_robustness=self.current_robustnesses[env_index]/self.current_lengths[env_index]
                self.mean_rewards.append(mean_reward)
                for i in range(len(mean_robustness)):
                    self.mean_robustnesses[i].append(mean_robustness[i])
                self.bool_terminations.append(self.locals["rewards"][env_index]>0)
                with open(self.rewards_path,mode='a',newline='') as file:
                    writer=csv.writer(file)
                    writer.writerow([self.tot_episodes,mean_reward])

                self.current_rewards[env_index]=0
                self.current_lengths [env_index]= 0
                self.current_robustnesses[env_index]=np.zeros(3)
                self.episodes[env_index]+=1
                self.tot_episodes+=1
        return True
    
    def on_training_end(self):
        plt.plot(self.mean_rewards)
        plt.xlabel("Episode")
        plt.ylabel("Mean Reward")
        plt.savefig(self.rewards_plot_path)
        plt.close()

        for i in range(3):
            plt.plot(self.mean_robustnesses[i])
            plt.xlabel("Episode")
            plt.ylabel("Mean robustness")
            plt.savefig(self.robustnesses_plot_paths[i])
            plt.close()

        plt.plot(self.bool_terminations)
        plt.xlabel("Episode")
        plt.ylabel("Boolean termination")
        plt.savefig(self.bools_plot_path)
        plt.close()