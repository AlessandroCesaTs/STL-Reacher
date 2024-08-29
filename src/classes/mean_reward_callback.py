import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common import base_class
from stable_baselines3.common.callbacks import BaseCallback
from simplemonitor.stl import STLMonitorBuilder, TimeSeries, RobSemantics

class MyCallback(BaseCallback):
    def __init__(self,rewards_path,plots_path):
        super().__init__()
        self.rewards_path=rewards_path
        self.rewards_plot_path=os.path.join(plots_path,'rewards.png')
        self.robustness_1_plot_path=os.path.join(plots_path,'robustness_1.png')
        self.robustness_2_plot_path=os.path.join(plots_path,'robustness_2.png')
        self.mean_rewards=[]
        stl_formula_1='t = 1000\n F_[0,t]((G1<0.2) & (F_[0,t]((G2<0.2)&(F_[0,t](G3<0.2)))))'
        stl_formula_2='t = 1000\n G_[0,t](!(A<0.2))'
        self.stl_robustnesses_1=[]
        self.stl_robustnesses_2=[]
        self.monitor_1=STLMonitorBuilder(stl_formula_1).build()
        self.monitor_2=STLMonitorBuilder(stl_formula_2).build()
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
        self.current_lengths = np.zeros(self.num_envs)
        self.num_of_goals=self.training_env.get_attr("num_of_goals")[0]
        self.num_of_avoids=self.training_env.get_attr("num_of_avoids")[0]
        self.episodes=np.zeros(self.num_envs)
        self.distances_from_goals=[[[] for _ in range (self.num_of_goals)] for _ in range (self.num_envs)]
        self.distances_from_avoids=[[[] for _ in range (self.num_of_avoids)] for _ in range (self.num_envs)]

    def _on_step(self):
        self.current_rewards+=self.locals['rewards'][0]
        self.current_lengths += 1
        for env_index in range(self.num_envs):
            for goal_index in range(self.num_of_goals):
                self.distances_from_goals[env_index][goal_index].append(self.locals['infos'][env_index]["Distances from goals"][goal_index])

            for avoid_index in range(self.num_of_avoids):
                self.distances_from_avoids[env_index][avoid_index].append(self.locals['infos'][env_index]["Distances from avoids"][avoid_index])
            """
            if self.current_lengths[env_index]>1:
                timeseries=self.get_time_series(env_index)
                self.stl_robustnesses_1.append(self.monitor_1.monitor(RobSemantics(timeSeries=timeseries,currentState=0)))
                self.stl_robustnesses_2.append(self.monitor_2.monitor(RobSemantics(timeSeries=timeseries,currentState=0)))
            """
            if self.locals['dones'][env_index]:
                mean_reward = self.current_rewards[env_index]/self.current_lengths[env_index]
                self.mean_rewards.append(mean_reward)
                with open(self.rewards_path,mode='a',newline='') as file:
                    writer=csv.writer(file)
                    writer.writerow([self.tot_episodes,mean_reward])

                self.current_rewards[env_index]=0
                self.current_lengths [env_index]= 0
                self.episodes[env_index]+=1
                self.tot_episodes+=1
                self.distances_from_goals[env_index]=[[] for _ in range (self.num_of_goals)]
                self.distances_from_avoids[env_index]=[[] for _ in range (self.num_of_avoids)] 
        return True
    
    def get_time_series(self,env_index):
        return TimeSeries(['G1','G2','G3','A'],np.arange(self.current_lengths[env_index]),np.concatenate((self.distances_from_goals[env_index],self.distances_from_avoids[env_index])))

    def on_training_end(self):
        plt.plot(self.mean_rewards)
        plt.xlabel("Episode")
        plt.ylabel("Mean Reward")
        plt.savefig(self.rewards_plot_path)
        plt.close()

        plt.plot(self.stl_robustnesses_1)
        plt.xlabel("Step")
        plt.ylabel("Robustness")
        plt.savefig(self.robustness_1_plot_path)
        plt.close()

        plt.plot(self.stl_robustnesses_2)
        plt.xlabel("Step")
        plt.ylabel("Robustness")
        plt.savefig(self.robustness_2_plot_path)
        plt.close()
