import os
import time
import matplotlib.pyplot as plt
from stable_baselines3.common.vec_env import VecEnv
from classes.my_callback import MyCallback
from classes.stl_evaluator import STLEvaluator

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
        self.plots_path=os.path.join(output_path,'plots')

        self.callback=MyCallback(rewards_path=rewards_path,plots_path=self.plots_path)

    def train(self,total_timesteps=81920):
        self.environment.reset()
        self.model.learn(total_timesteps=total_timesteps,callback=self.callback,progress_bar=True)
        self.model.save(self.model_path)

    def test(self,max_test_steps=200,test_runs=1):
        for i in range(test_runs):
            if i==test_runs-1:
                self.environment.env_method("enable_video_mode",indices=0) if self.is_vectorized_environment else self.environment.enable_video_mode()
            test_rewards=[]
            observation,info=self.environment.env_method('reset',indices=0)[0] if self.is_vectorized_environment else self.environment.reset()[0]
            terminated=False
            truncated=False
            test_steps=0
            while not (terminated or truncated):
                action,_states=self.model.predict(observation)
                observation, reward, terminated, truncated, info = self.environment.env_method('step',action,indices=0)[0] if self.is_vectorized_environment else self.environment.step(action)
                test_rewards.append(reward)
                test_steps+=1
                if test_steps>max_test_steps:
                    break
            if i==test_runs-1:
                if self.is_vectorized_environment:
                    print("saving video")
                    self.environment.env_method('save_video',indices=0)
                else:
                    self.environment.save_video()

            plt.plot(test_rewards)
            plt.xlabel("Step")
            plt.ylabel("Reward")
            path=os.path.join(self.plots_path,'test_rewards_'+str(i)+'png')
            plt.savefig(path)
            plt.close()





    

