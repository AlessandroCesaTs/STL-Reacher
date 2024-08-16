import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback


class MeanRewardCallback(BaseCallback):
    def __init__(self,verbose=0):
        super(MeanRewardCallback,self).__init__(verbose)
        self.mean_rewards=[]
        self.current_rewards=0
        self.episode_length = 0


    def _on_step(self):
        self.current_rewards+=self.locals['rewards'][0]
        self.episode_length += 1
        if self.locals['dones'][0]:
            mean_reward = self.current_rewards/self.episode_length
            self.mean_rewards.append(mean_reward)
            if self.verbose > 0:
                print(f"Episode {len(self.mean_rewards)}: mean reward = {mean_reward}")

            self.current_rewards=0
            self.episode_length = 0
        return True

    def on_training_end(self):
        # Save the mean rewards to a file or process them here
        plt.plot(self.mean_rewards)
        plt.show()
        plt.close()
        if self.verbose > 0:
            print("Mean rewards saved.")