import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common import base_class
from stable_baselines3.common.callbacks import BaseCallback
from classes.stl_evaluator import STLEvaluator
class MyCallback(BaseCallback):

    def __init__(self,rewards_path,plots_path):
        super().__init__()
        self.rewards_path=rewards_path
        self.rewards_plot_path=os.path.join(plots_path,'rewards.png')
        self.robustness_1_plot_path=os.path.join(plots_path,'robustnesses.png')
        self.mean_rewards=[]
        self.mean_robustnesses=[]
        self.tot_episodes=0
        self.formula=["and",["F", ["and", 0, ["F", ["and", 1, ["F", 2]]]]],3]
        
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
        self.current_robustness=np.zeros(self.num_envs)
        self.current_lengths = np.zeros(self.num_envs)
        self.num_of_signals=self.training_env.get_attr("num_of_goals",indices=0)[0]+self.training_env.get_attr("num_of_avoids",indices=0)[0]
        self.episodes=np.zeros(self.num_envs)
        self.signals=[[[] for _ in range (self.num_of_signals)] for _ in range (self.num_envs)]
        self.evaluators=[STLEvaluator(self.signals[i],self.formula) for i in range(self.num_envs)]
        self.formulas=[evaluator.apply_formula() for evaluator in self.evaluators]

    def _on_step(self):
        self.current_rewards+=self.locals['rewards'][0]
        self.current_lengths += 1
        for env_index in range(self.num_envs):
            for signal_index in range(self.num_of_signals):
                self.evaluators[env_index].append_signal(signal_index,self.locals['infos'][env_index]["Signals"][signal_index].item())
            
            self.current_robustness[env_index]+=self.formulas[env_index](0)
            
            if self.locals['dones'][env_index]:
                mean_reward = self.current_rewards[env_index]/self.current_lengths[env_index]
                mean_robustness=self.current_robustness[env_index]/self.current_lengths[env_index]
                self.mean_rewards.append(mean_reward)
                self.mean_robustnesses.append(mean_robustness)
                with open(self.rewards_path,mode='a',newline='') as file:
                    writer=csv.writer(file)
                    writer.writerow([self.tot_episodes,mean_reward])

                self.current_rewards[env_index]=0
                self.current_lengths [env_index]= 0
                self.current_robustness[env_index]=0
                self.episodes[env_index]+=1
                self.tot_episodes+=1
                self.signals[env_index]=[[] for _ in range (self.num_of_signals)]
        return True
    
    def on_training_end(self):
        plt.plot(self.mean_rewards)
        plt.xlabel("Episode")
        plt.ylabel("Mean Reward")
        plt.savefig(self.rewards_plot_path)
        plt.close()

        plt.plot(self.mean_robustnesses)
        plt.xlabel("Episode")
        plt.ylabel("Mean Robustness")
        plt.savefig(self.robustness_1_plot_path)
        plt.close()


