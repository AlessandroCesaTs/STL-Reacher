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
        train_logs_path=os.path.join(self.output_path,'training','logs')
        model_path=os.path.join(self.output_path,'model')
        os.makedirs(train_logs_path, exist_ok=True)

        callback=MyCallback(train_logs_path)

        self.environment.reset()
        self.model.learn(total_timesteps=total_timesteps,callback=callback,progress_bar=True)
        self.model.save(model_path)

    def test(self,test_runs=1):
        if self.is_vectorized_environment:
            self.environment.env_method("enable_video_mode",indices=0)
            num_of_formulas=self.environment.get_attr('number_of_formulas',indices=0)
        else:
            self.environment.enable_video_mode()
            num_of_formulas=self.environment.number_of_formulas
        
        test_logs_path=os.path.join(self.output_path,'test','logs')
        os.makedirs(test_logs_path, exist_ok=True)
        
        rewards_log_path=os.path.join(test_logs_path,'test_rewards.csv')
        robustnesses_log_path=[os.path.join(test_logs_path,f"test_robustnesses_{i}.csv") for i in range(num_of_formulas)]
        final_robustness_log_path=os.path.join(test_logs_path,'final_robustness.csv')
        final_boolean_log_path=os.path.join(test_logs_path,'final_boolean.csv')

        write_to_csv(rewards_log_path,['Run','Step','Reward'],'w')
        write_to_csv(final_robustness_log_path,['Run','Robustness'],'w')
        write_to_csv(final_boolean_log_path,['Run','Boolean'],'w')
        for robustness_index in range(num_of_formulas):
             write_to_csv(robustnesses_log_path[robustness_index],['Run','Step','Robustness'],'w')
        

        for run in range(test_runs):
            observation,info=self.environment.env_method('reset',indices=0)[0] if self.is_vectorized_environment else self.environment.reset()[0]
            terminated=False
            truncated=False

            while not (terminated or truncated):
                action,_states=self.model.predict(observation)
                observation, reward, terminated, truncated, info = self.environment.env_method('step',action,indices=0)[0] if self.is_vectorized_environment else self.environment.step(action)
                step=info['step']
                goal_to_reach=info['goal_to_reach']

                write_to_csv(rewards_log_path,[run,step,reward],'a')
                write_to_csv(robustnesses_log_path[goal_to_reach],[run,step,reward])
                
            final_robustness=info['final_robustness']
            final_boolean=info['final_boolean']
            write_to_csv(self.final_robustness_log_path,[run,final_robustness],'a')
            write_to_csv(self.final_boolean_log_path,[run,final_boolean],'a')

            if self.is_vectorized_environment:
                self.environment.env_method('save_video',f"_{run}",indices=0)
            else:
                self.environment.save_video(f"_{run}")    

