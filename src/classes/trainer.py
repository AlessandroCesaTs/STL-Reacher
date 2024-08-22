import os
import time
import matplotlib.pyplot as plt
from stable_baselines3.common.vec_env import VecEnv
from classes.mean_reward_callback import MeanRewardCallback

class Trainer:
    def __init__(self,environment,model,output_path):
        self.environment=environment
        self.model=model
        self.is_vectorized_environment=isinstance(self.environment,VecEnv)

        os.makedirs(os.path.join(output_path,'models'), exist_ok=True)
        os.makedirs(os.path.join(output_path,'rewards'), exist_ok=True)
        os.makedirs(os.path.join(output_path,'plots'), exist_ok=True)

        self.model_path=os.path.join(output_path,'models','model')
        rewards_path=os.path.join(output_path,'rewards','rewards.csv')
        plots_path=os.path.join(output_path,'plots')
        train_plot_path=os.path.join(plots_path,'train_rewards.png')
        self.test_plot_path=os.path.join(plots_path,'test_rewards.png')

        self.callback=MeanRewardCallback(rewards_path=rewards_path,plots_path=train_plot_path)

    def train(self,total_timesteps=81920):
        self.environment.reset()
        start_time=time.time()
        self.model.learn(total_timesteps=total_timesteps,callback=self.callback)
        print(f"Total Time: {(time.time()-start_time)/60} minutes")
        self.model.save(self.model_path)

    def test(self,test_steps=200):
        test_rewards=[]
        if self.is_vectorized_environment:
            observation,info=self.environment.env_method('reset',indices=0)[0] #env_method returns list
            self.environment.env_method("enable_video_mode",indices=0)
            
        else:
            observation=self.environment.reset()[0]
            self.environment.enable_video_mode()
        for i in range(test_steps):
            action,_states=self.model.predict(observation)
            observation, reward, terminated, truncated, info = self.environment.env_method('step',action,indices=0)[0] if self.is_vectorized_environment else self.environment.step(action)
            test_rewards.append(reward)
            if terminated or truncated:
                break
        if self.is_vectorized_environment:
            self.environment.env_method('save_video',indices=0)
        else:
            self.environment.save_video()

        plt.plot(test_rewards)
        plt.xlabel("Step")
        plt.ylabel("Reward")
        plt.savefig(self.test_plot_path)
        plt.close()

        







    

