import os
from stable_baselines3.common.vec_env import VecEnv
from classes.my_callback import MyCallback
from utils.utils import write_to_csv

class Trainer:
    def __init__(self,environment,model,output_path):
        self.environment=environment
        self.model=model
        self.output_path=output_path
        self.is_vectorized_environment=isinstance(self.environment,VecEnv)

    def train(self,total_timesteps=81920):
        train_logs_path=os.path.join(self.output_path,'train','logs')
        model_path=os.path.join(self.output_path,'model')
        os.makedirs(train_logs_path, exist_ok=True)

        callback=MyCallback(train_logs_path)

        self.environment.reset()

        self.model.learn(total_timesteps=total_timesteps,callback=callback)
        self.model.save(model_path)

    
    def test_single_target(self,test_runs=1):
        if self.is_vectorized_environment:
            self.environment.env_method("enable_video_mode",indices=0)
            self.environment.env_method("set_start_goal_avoid_from_file",indices=0)
        else:
            self.environment.enable_video_mode()
            self.environment.set_start_goal_avoid_from_file()
        
        test_logs_path=os.path.join(self.output_path,'test','logs')
        videos_path=os.path.join(self.output_path,'test','videos')
        os.makedirs(test_logs_path, exist_ok=True)
        os.makedirs(videos_path, exist_ok=True)

        rewards_log_path=os.path.join(test_logs_path,'rewards.csv')
        final_state_log_path=os.path.join(test_logs_path,'final_state.csv')

        write_to_csv(rewards_log_path,['Run','Step','Reward'],'w')
        write_to_csv(final_state_log_path,['Run','Final_State'],'w')

        for run in range(test_runs):
            observation,info=self.environment.env_method('reset',indices=0)[0] if self.is_vectorized_environment else self.environment.reset()[0]
            terminated=False
            truncated=False

            while not terminated or truncated:
                action,_states=self.model.predict(observation)
                observation, reward, terminated, truncated, info = self.environment.env_method('step',action,indices=0)[0] if self.is_vectorized_environment else self.environment.step(action)
                step=info['step']

                write_to_csv(rewards_log_path,[run,step,reward],'a')

                if terminated or truncated:
                    final_state=info['final_state']
                    write_to_csv(final_state_log_path,[run,final_state],'w')

                
            video_path=os.path.join(videos_path,f"video_{run}.avi")
            if self.is_vectorized_environment:
                self.environment.env_method('save_video',video_path,indices=0)
            else:
                self.environment.save_video(video_path)    






