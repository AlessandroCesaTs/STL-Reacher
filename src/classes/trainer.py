import os
from stable_baselines3.common.vec_env import VecEnv
from classes.my_callback import MyCallback
from utils.utils import write_to_csv
from classes.stl_evaluator import STLEvaluator
import numpy as np

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
        setting_path=os.path.join(self.output_path,'setting.pkl')
        if self.is_vectorized_environment:
            self.environment.env_method("enable_video_mode",indices=0)
            self.environment.env_method("set_setting_from_file",setting_path,indices=0)
        else:
            self.environment.enable_video_mode()
            self.environment.set_setting_from_file(setting_path)
        
        test_logs_path=os.path.join(self.output_path,'test','logs')
        videos_path=os.path.join(self.output_path,'test','videos')
        os.makedirs(test_logs_path, exist_ok=True)
        os.makedirs(videos_path, exist_ok=True)

        end_conditions_log_path=os.path.join(test_logs_path,'end_conditions.csv')

        write_to_csv(end_conditions_log_path,['Run','Robustness','End_Condition'],'w')

        for run in range(test_runs):
            observation,info=self.environment.env_method('reset',indices=0)[0] if self.is_vectorized_environment else self.environment.reset()[0]
            terminated=False
            truncated=False

            while not (terminated or truncated):
                action,_states=self.model.predict(observation)
                observation, reward, terminated, truncated, info = self.environment.env_method('step',action,indices=0)[0] if self.is_vectorized_environment else self.environment.step(action)

                if terminated or truncated:
                    robustness=info['requirement_robustness']
                    end_condition=info['end_condition']
                    write_to_csv(end_conditions_log_path,[run,robustness,end_condition],'a')

                
            video_path=os.path.join(videos_path,f"video_{run}.avi")
            if self.is_vectorized_environment:
                self.environment.env_method('save_video',video_path,indices=0)
            else:
                self.environment.save_video(video_path)

    def test_changing_target(self,test_runs=1):
        if self.is_vectorized_environment:
            self.environment.env_method("enable_video_mode",indices=0)
        else:
            self.environment.enable_video_mode()
        
        test_logs_path=os.path.join(self.output_path,'test','logs')
        videos_path=os.path.join(self.output_path,'test','videos')
        os.makedirs(test_logs_path, exist_ok=True)
        os.makedirs(videos_path, exist_ok=True)

        end_conditions_log_path=os.path.join(test_logs_path,'end_conditions.csv')

        write_to_csv(end_conditions_log_path,['Run','Robustness','End_Condition'],'w')

        for run in range(test_runs):
            observation,info=self.environment.env_method('reset',indices=0)[0] if self.is_vectorized_environment else self.environment.reset()[0]
            setting_1=self.environment.env_method('get_setting',indices=0)[0] if self.is_vectorized_environment else self.environment.get_setting()
            
            observation,info=self.environment.env_method('reset',indices=0)[0] if self.is_vectorized_environment else self.environment.reset()[0]
            setting_0=self.environment.env_method('get_setting',indices=0)[0] if self.is_vectorized_environment else self.environment.get_setting()

            goal_0=setting_0['goal']
            avoid_0=setting_0['avoid']

            goal_1=setting_1['goal']
            avoid_1=setting_1['avoid']

            signals=[[],[],[],[]]

            first_part=["and",0,["G",2]]
            second_part=["and",1,["G",3]]
            requirement_formula=["F",["and",first_part,["F",second_part]]]

            terminated=False
            truncated=False

            goal=0

            while not (terminated or truncated):
                action,_states=self.model.predict(observation)
                observation, reward, terminated, truncated, info = self.environment.env_method('step',action,indices=0)[0] if self.is_vectorized_environment else self.environment.step(action)

                end_effector_position=info['end_effector_position']

                signals[0].append(np.linalg.norm(goal_0-end_effector_position))
                signals[1].append(np.linalg.norm(goal_1-end_effector_position))
                signals[2].append(np.linalg.norm(avoid_0-end_effector_position))
                signals[3].append(np.linalg.norm(avoid_1-end_effector_position))

                if goal==0 and reward>0:
                    self.environment.env_method('set_setting',setting_1,indices=0) if self.is_vectorized_environment else self.environment.set_setting(setting_1)
                    goal+=1
                elif goal==1 and reward>0:
                    terminated=True

                
            requirement_evaluator=STLEvaluator(signals,requirement_formula)

            requirement_evaluating_function=requirement_evaluator.apply_formula()

            robustness=requirement_evaluating_function(0)
            end_condition='reach_stay_no_collision' if robustness>0 else 'no_reach_collision' 

            write_to_csv(end_conditions_log_path,[run,robustness,end_condition],'a')
                
            video_path=os.path.join(videos_path,f"video_{run}.avi")
            if self.is_vectorized_environment:
                self.environment.env_method('save_video',video_path,indices=0)
            else:
                self.environment.save_video(video_path)    






