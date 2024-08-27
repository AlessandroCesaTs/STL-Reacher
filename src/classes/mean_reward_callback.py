import csv
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common import base_class
from stable_baselines3.common.callbacks import BaseCallback

class MeanRewardCallback(BaseCallback):
    def __init__(self,rewards_path,plots_path):
        super().__init__()
        self.rewards_path=rewards_path
        self.plots_path=plots_path
        self.mean_rewards=[]
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
        self.episodes=np.zeros(self.num_envs)

    def _on_step(self):
        self.current_rewards+=self.locals['rewards']
        self.current_lengths += 1
        for i in range(self.num_envs):
            if self.locals['dones'][i]:
                mean_reward = self.current_rewards[i]/self.current_lengths[i]
                self.mean_rewards.append(mean_reward)
                with open(self.rewards_path,mode='a',newline='') as file:
                    writer=csv.writer(file)
                    writer.writerow([self.tot_episodes,mean_reward])

                self.current_rewards[i]=0
                self.current_lengths [i]= 0
                self.episodes[i]+=1
                self.tot_episodes+=1
        return True

    def on_training_end(self):
        plt.plot(self.mean_rewards)
        plt.xlabel("Episode")
        plt.ylabel("Mean Reward")
        plt.savefig(self.plots_path)
        plt.close()
        print(f"Number of episodes {self.tot_episodes}")
