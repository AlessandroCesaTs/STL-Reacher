import csv
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback

class MeanRewardCallback(BaseCallback):
    def __init__(self,rewards_path,plots_path):
        super(MeanRewardCallback,self).__init__()
        self.rewards_path=rewards_path
        self.plots_path=plots_path
        self.mean_rewards=[]
        self.mean_rewards=[]
        self.current_rewards=0
        self.episode_length = 0
        self.episode=1

        with open(self.rewards_path,mode='w',newline='') as file:
            writer=csv.writer(file)
            writer.writerow(['Episode','Mean Reward'])


    def _on_step(self):
        self.current_rewards+=self.locals['rewards'][0]
        self.episode_length += 1
        if self.locals['dones'][0]:
            mean_reward = self.current_rewards/self.episode_length
            self.mean_rewards.append(mean_reward)
            with open(self.rewards_path,mode='a',newline='') as file:
                writer=csv.writer(file)
                writer.writerow([self.episode,mean_reward])

            self.current_rewards=0
            self.episode_length = 0
            self.episode+=1
        return True

    def on_training_end(self):
        plt.plot(self.mean_rewards)
        plt.xlabel("Episode")
        plt.ylabel("Mean Reward")
        plt.savefig(self.plots_path)
        plt.close()
