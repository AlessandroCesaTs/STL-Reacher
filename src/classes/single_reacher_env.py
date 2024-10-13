import os
import numpy as np
import gymnasium as gym
import pybullet as p
import cv2
from gymnasium import spaces
from gym_ergojr.sim.single_robot import SingleRobot
from gym_ergojr.sim.objects import Ball
from gym_ergojr.utils.math import RandomPointInHalfSphere
from classes.stl_evaluator import STLEvaluator
import pickle
from utils.utils import copy_urdf_directory

urdf_default_dir='env/lib/python3.12/site-packages/gym_ergojr/scenes/'

class SingleReacherEnv(gym.Env):
    def __init__(self,urdf_dir=urdf_default_dir,max_steps=100,output_path=os.getcwd(), hard_reward=False):
        super().__init__()
        self.observation_space = spaces.Box(low=-1, high=1, shape=(9,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32)
        self.output_path=output_path

        self.video_mode=False

        self.urdf_dir=copy_urdf_directory(urdf_dir)
        self.rhis = RandomPointInHalfSphere(0,0,0,radius=0.2,min_dist=0.1)

        self.robot = SingleRobot(urdf_dir=self.urdf_dir)

        self.sphere_radius = 0.02  # distance between robot tip and goal under which the task is considered solved
        self.necessary_distance= self.sphere_radius/2
        self.min_distance=0.1
        self.max_iterations_to_check_point=500

        self.steps=0
        self.start_computing_robustness_from=0
        self.max_steps=max_steps  
        self.episodes=0
        

        self.hard_reward=hard_reward

        
        self.new_start_goal_avoid()


        signals=[[],[],[]]

        self.reach_formula=["F",0]
        self.stay_formula=["F",["G",0]]
        self.collision_formula=["G",1]
        self.requirement=["and",self.stay_formula,self.collision_formula]
        self.hard_reward_formula=["and",self.stay_formula,["G",2]]

        self.evaluator=STLEvaluator(signals,self.requirement)
        self.reach_evaluator=STLEvaluator(signals,self.reach_formula)
        self.stay_evaluator=STLEvaluator(signals,self.stay_formula)
        self.collision_evaluator=STLEvaluator(signals,self.collision_formula)
        self.hard_reward_evaluator=STLEvaluator(signals,self.hard_reward_formula)

        self.evaluating_function=self.evaluator.apply_formula()
        self.reach_evaluating_function=self.reach_evaluator.apply_formula()
        self.stay_evaluating_function=self.stay_evaluator.apply_formula()
        self.collision_evaluating_function=self.collision_evaluator.apply_formula()
        self.hard_reward_evaluating_function=self.hard_reward_evaluator.apply_formula()
    

    def set_setting(self,setting):
        self.starting_point=setting['starting_point']
        self.initial_pose=setting['initial_pose']
        self.goal=setting['goal']
        self.avoid=setting['avoid']
        #self.flatten_points=np.concatenate([self.goal,self.avoid])
        self.robot.set(self.initial_pose)

        if self.video_mode:
            self.set_and_move_graphic_balls()
    
    def set_setting_from_file(self,file):
        with open(file, 'rb') as f:
            setting = pickle.load(f)
        self.starting_point=setting['starting_point']
        self.initial_pose=setting['initial_pose']
        self.goal=setting['goal']
        self.avoid=setting['avoid']
        #self.flatten_points=np.concatenate([self.goal,self.avoid])
        self.robot.set(self.initial_pose)

        if self.video_mode:
            self.set_and_move_graphic_balls()


    def new_start_goal_avoid(self):

        self.starting_point,self.initial_pose=self.get_reachable_point()
        
        points_found=False
        while not points_found:
            self.goal,_=self.get_reachable_point()
            start_goal_distance=np.linalg.norm(self.starting_point - self.goal)
            if start_goal_distance<self.min_distance:
                points_found=False
            else:
                alpha_min=self.min_distance/start_goal_distance
                alpha_max=1-alpha_min
                alpha =  np.random.uniform(alpha_min, alpha_max)
                self.avoid=(1-alpha)*self.starting_point+alpha*self.goal
                avoid_is_reachable,_=self.is_reachable(self.avoid)
                if avoid_is_reachable:
                    points_found=True
                else:
                    points_found=False

        #self.flatten_points=np.concatenate([self.goal,self.avoid])

        if self.video_mode:
            self.set_and_move_graphic_balls()
        
        self.robot.set(self.initial_pose)

    def save_setting_to_file(self,setting_path):
        setting={'starting_point':self.starting_point, 'initial_pose':self.initial_pose,'goal':self.goal,'avoid':self.avoid}
        with open(setting_path,'wb') as f:
            pickle.dump(setting,f)
    def get_setting(self):
        setting={'starting_point':self.starting_point, 'initial_pose':self.initial_pose,'goal':self.goal,'avoid':self.avoid}
        return setting
    
    def reset(self,**kwargs):

        
        self.steps=0
        
        self.robot.set(self.initial_pose)

        self.evaluator.reset_signals()
        self.reach_evaluator.reset_signals()
        self.stay_evaluator.reset_signals()
        self.collision_evaluator.reset_signals()
        self.hard_reward_evaluator.reset_signals()

        self.evaluating_function=self.evaluator.apply_formula()
        self.reach_evaluating_function=self.reach_evaluator.apply_formula()
        self.stay_evaluating_function=self.stay_evaluator.apply_formula()
        self.collision_evaluating_function=self.collision_evaluator.apply_formula()
        self.hard_reward_evaluating_function=self.hard_reward_evaluator.apply_formula()

        observation=self._get_obs()
        reset_info={} #needed for stable baseline

        if self.video_mode:
            self.frames=[]

        return observation,reset_info
    
    def set_and_move_graphic_balls(self):
        self.goal_ball.changePos(self.goal, 4)
        self.avoid_ball.changePos(self.avoid, 4)

        for _ in range(25):
            self.robot.step()

    def step(self, action):
        action=np.array(action)

        self.robot.act2(action)
        self.robot.step()

        reward, terminated, truncated, info = self._getReward()

        obs = self._get_obs()

        self.steps+=1
        
        if self.video_mode:
            image=self._capture_image()
            self.frames.append(image)
        
        return obs,reward,terminated,truncated,info
    
    def _getReward(self):
        terminated=False
        truncated = False

        distance_from_goal=self.distance_from_goal()
        distance_from_avoid=self.distance_from_avoid()
        
        goal_signal=self.sphere_radius-distance_from_goal
        avoid_collision_signal=distance_from_avoid-self.sphere_radius
        avoid_near_signal=1.5*(distance_from_avoid-2*self.sphere_radius)

        signals=np.array([goal_signal,avoid_collision_signal,avoid_near_signal])
        self.evaluator.append_signals(signals)
        self.reach_evaluator.append_signals(signals)
        self.stay_evaluator.append_signals(signals)
        self.collision_evaluator.append_signals(signals)
        self.hard_reward_evaluator.append_signals(signals)

        requirement_robustness=self.evaluating_function(0)

        if self.hard_reward:
            reward=self.hard_reward_evaluating_function(0)
        else:
            reward=requirement_robustness
        
        info={'episode_number':self.episodes,'step':self.steps,'requirement_robustness':requirement_robustness,'end_effector_position':self.get_position_of_end_effector()}            
        
        if self.steps>self.max_steps:
            terminated=True
            if self.reach_evaluating_function(0)>0:
                if self.collision_evaluating_function(0)>0:
                    if self.stay_evaluating_function(0)>0:
                        end_condition='reach_stay_no_collision'
                    else:
                        end_condition='reach_no_stay_no_collision'
                else:
                    if self.stay_evaluating_function(0)>0:
                        end_condition='reach_stay_collision'
                    else:
                        end_condition='reach_no_stay_collision'
            else:
                if self.collision_evaluating_function(0)>0:
                    end_condition='no_reach_no_collision'
                else:
                    end_condition='no_reach_collision'

        if terminated or truncated:
            info['end_condition']=end_condition
            self.episodes+=1

        return reward, terminated, truncated, info

    def _capture_image(self):
        view_matrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[0, 0, 0],
                                                          distance=1.5,
                                                          yaw=50,
                                                          pitch=-35,
                                                          roll=0,
                                                          upAxisIndex=2)
        
        proj_matrix = p.computeProjectionMatrixFOV(fov=30,
                                                   aspect=float(self.image_size[0]) / self.image_size[1],
                                                   nearVal=0.1,
                                                   farVal=100.0)
        (_, _, px, _, _) = p.getCameraImage(width=self.image_size[0],
                                            height=self.image_size[1],
                                            viewMatrix=view_matrix,
                                            projectionMatrix=proj_matrix,
                                            renderer=p.ER_BULLET_HARDWARE_OPENGL)
        rgb_array = np.array(px)
        image_array=rgb_array.reshape((self.image_size[1],self.image_size[0],4))
        image_array=image_array[:, :, :3] #remove alpha channel
        image_array = image_array.astype(np.uint8) #convert to correct type for CV2
        return image_array 

    def save_video(self,path):
        if self.frames:
            out=cv2.VideoWriter(path,cv2.VideoWriter_fourcc(*'XVID'),fps=self.fps,frameSize=self.image_size)
            for image in self.frames:
                out.write(image)
            out.release()      

    def enable_video_mode(self):
        self.frames=[]
        self.image_size=(640,480)
        self.fps=5
        self.goal_ball=Ball(self.urdf_dir,color="green")
        self.avoid_ball=Ball(self.urdf_dir,color="red")
        self.video_mode=True

    def disable_video_mode(self):
        self.video_mode=False

    def _get_obs(self):
        observation=self.robot.observe()[:6]
        position_of_end_effector=self.get_position_of_end_effector()
        obs=np.concatenate([observation,position_of_end_effector])
        return obs
    
    def close(self):
        self.robot.close()
    def distance_from_goal(self):
        return np.linalg.norm(self.goal-self.get_position_of_end_effector())
    def distance_from_goal_1(self):
        return np.linalg.norm(self.goal_1-self.get_position_of_end_effector())
    def distance_from_goal_2(self):
        return np.linalg.norm(self.goal_2-self.get_position_of_end_effector())
    def distance_from_avoid(self):
        return np.linalg.norm(self.avoid-self.get_position_of_end_effector())
    def distance_from_avoid_1(self):
        return np.linalg.norm(self.avoid_1-self.get_position_of_end_effector())
    def distance_from_avoid_2(self):
        return np.linalg.norm(self.avoid_2-self.get_position_of_end_effector())
    def get_position_of_end_effector(self):
        return np.array(p.getLinkState(self.robot.id,13)[0])

    def get_reachable_point(self):
        reachable=False
        while not reachable:
            point=self.rhis.samplePoint()
            reachable, joints_positions_velocities = self.is_reachable(point)
        return point,joints_positions_velocities

    def is_reachable(self, point):
        distance=np.inf
        iter=0
        while distance>self.necessary_distance and iter<self.max_iterations_to_check_point:
            joints_positions=p.calculateInverseKinematics(bodyIndex=1, 
                                        endEffectorLinkIndex=13, 
                                        targetPosition=point)
            joints_positions_velocities=np.concatenate([np.array(joints_positions),np.zeros(5)])
            self.robot.set(joints_positions_velocities)
            end_effector_position=self.get_position_of_end_effector()
            distance=np.linalg.norm(end_effector_position-point)
            iter+=1
        if iter!=self.max_iterations_to_check_point:
            reachable=True
        else:
            reachable=False
        return reachable,joints_positions_velocities
    
        

