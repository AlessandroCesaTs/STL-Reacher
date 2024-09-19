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
        self.model.learn(total_timesteps=total_timesteps,callback=callback,progress_bar=True)
        self.model.save(model_path)

    def test(self,test_runs=1):
        if self.is_vectorized_environment:
            self.environment.env_method("enable_video_mode",indices=0)
            num_of_formulas_to_monitor=self.environment.get_attr('number_of_formulas_to_monitor',indices=0)[0]
        else:
            self.environment.enable_video_mode()
            num_of_formulas_to_monitor=self.environment.number_of_formulas_to_monitor
        
        test_logs_path=os.path.join(self.output_path,'test','logs')
        os.makedirs(test_logs_path, exist_ok=True)
        
        rewards_log_path=os.path.join(test_logs_path,'rewards.csv')
        safeties_log_path=os.path.join(test_logs_path,'safeties.csv')
        dist_log_path=os.path.join(test_logs_path,'dist.csv')
        robustnesses_log_path=[os.path.join(test_logs_path,f"robustness_{i}.csv") for i in range(num_of_formulas_to_monitor)]
        final_robustness_log_path=os.path.join(test_logs_path,'final_robustness.csv')
        final_boolean_log_path=os.path.join(test_logs_path,'final_boolean.csv')
        videos_path=os.path.join(self.output_path,'test','videos')
        os.makedirs(videos_path, exist_ok=True)

        write_to_csv(rewards_log_path,['Run','Step','Reward'],'w')
        write_to_csv(safeties_log_path,['Run','Step','Robustness'],'w')
        write_to_csv(dist_log_path,['Run','Step','Distances'],'w')
        write_to_csv(final_robustness_log_path,['Run','Robustness'],'w')
        write_to_csv(final_boolean_log_path,['Run','Boolean'],'w')
        for robustness_index in range(num_of_formulas_to_monitor):
             write_to_csv(robustnesses_log_path[robustness_index],['Run','Step','Robustness'],'w')
        

        for run in range(test_runs):
            observation,info=self.environment.env_method('reset',indices=0)[0] if self.is_vectorized_environment else self.environment.reset()[0]
            terminated=False
            truncated=False

            while not (terminated or truncated):
                action,_states=self.model.predict(observation)
                observation, reward, terminated, truncated, info = self.environment.env_method('step',action,indices=0)[0] if self.is_vectorized_environment else self.environment.step(action)
                step=info['step']
                safety=info['safety']
                distances=info['distances']
                goal_to_reach=info['goal_to_reach']

                write_to_csv(rewards_log_path,[run,step,reward],'a')
                write_to_csv(safeties_log_path,[run,step,safety],'a')
                write_to_csv(dist_log_path,[run,step,distances],'a')
                write_to_csv(robustnesses_log_path[goal_to_reach],[run,step,reward],'a')
                
            final_robustness=info['final_robustness']
            final_boolean=info['final_boolean']
            write_to_csv(final_robustness_log_path,[run,final_robustness],'a')
            write_to_csv(final_boolean_log_path,[run,final_boolean],'a')

            video_path=os.path.join(videos_path,f"video_{run}.avi")
            if self.is_vectorized_environment:
                self.environment.env_method('save_video',video_path,indices=0)
            else:
                self.environment.save_video(video_path)    

