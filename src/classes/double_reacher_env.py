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

class DoubleReacherEnv(gym.Env):
    def __init__(self,urdf_dir=urdf_default_dir,max_steps=500,output_path=os.getcwd()):
        super().__init__()
        self.observation_space = spaces.Box(low=-1, high=1, shape=(10,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32)
        self.output_path=output_path

        self.video_mode=False

        self.urdf_dir=copy_urdf_directory(urdf_dir)
        self.rhis = RandomPointInHalfSphere(0,0,0,radius=0.2,min_dist=0.1)

        self.robot = SingleRobot(urdf_dir=self.urdf_dir)

        self.first_complete=False

        self.sphere_radius = 0.02  # distance between robot tip and goal under which the task is considered solved
        self.hard_sphere_radius=2*self.sphere_radius
        self.necessary_distance= self.sphere_radius/2
        self.min_distance=0.1
        self.max_iterations_to_check_point=500

        self.steps=0
        self.start_computing_robustness_from=0
        self.max_steps=max_steps  
        self.episodes=0
        self.new_start_goal_avoid()

        self.goal_index=0
        self.robot.set(self.initial_pose)

        signals=[[],[],[],[],[],[]]

        reach_1_formula=["F",["G",0]]
        reach_2_formula=["F",["G",3]]

        eventually_reach_1_and_eventually_reach_2_formula=["F",["and",0,["F",["G",3]]]]
        globally_avoid_formula=["G",["and",1,4]]
        globally_avoid_hard_formula=["G",["and",2,5]]

        first_part_formula=["and",reach_1_formula,globally_avoid_formula]
        second_part_formula=["and",reach_2_formula,globally_avoid_formula]
        first_part_hard_formula=["and",reach_1_formula,globally_avoid_hard_formula]
        second_part_hard_formula=["and",reach_2_formula,globally_avoid_hard_formula]

        requirement_formula=["and",eventually_reach_1_and_eventually_reach_2_formula,globally_avoid_formula]
        reward_formula=["and",eventually_reach_1_and_eventually_reach_2_formula,globally_avoid_hard_formula]

        self.requirement_evaluator=STLEvaluator(signals,requirement_formula)
        self.requirement_evaluating_function=self.requirement_evaluator.apply_formula()
        
        #self.reward_evaluator=STLEvaluator(signals,reward_formula)
        #self.reward_evaluating_function=self.reward_evaluator.apply_formula()
        
        self.first_part_evaluator=STLEvaluator(signals,first_part_formula)
        self.first_part_evaluating_function=self.first_part_evaluator.apply_formula()

        self.second_part_evaluator=STLEvaluator(signals,second_part_formula)
        self.second_part_evaluating_function=self.second_part_evaluator.apply_formula()

        #self.first_part_hard_evaluator=STLEvaluator(signals,first_part_hard_formula)
        #self.first_part_hard_evaluating_function=self.first_part_hard_evaluator.apply_formula()

        #self.second_part_hard_evaluator=STLEvaluator(signals,second_part_hard_formula)
        #self.second_part_hard_evaluating_function=self.second_part_hard_evaluator.apply_formula()

        

    def set_setting_from_file(self,file):
        with open(file, 'rb') as f:
            setting = pickle.load(f)
        self.initial_pose=setting['initial_pose']
        self.goal_1=setting['goal_1']
        self.avoid_1=setting['avoid_1']
        self.goal_2=setting['goal_2']
        self.avoid_2=setting['avoid_2']
        self.robot.set(self.initial_pose)

        if self.video_mode:
            self.set_and_move_graphic_balls()


    def new_start_goal_avoid(self):
        starting_point,self.initial_pose=self.get_reachable_point()
        
        points_found=False
        while not points_found:
            self.goal_1,self.second_initial_pose=self.get_reachable_point()
            start_goal_distance=np.linalg.norm(starting_point - self.goal_1)
            if start_goal_distance<self.min_distance:
                points_found=False
            else:
                alpha_min=self.min_distance/start_goal_distance
                alpha_max=1-alpha_min
                alpha =  np.random.uniform(alpha_min, alpha_max)
                self.avoid_1=(1-alpha)*starting_point+alpha*self.goal_1
                avoid_is_reachable,_=self.is_reachable(self.avoid_1)
                if avoid_is_reachable:
                    self.goal_2,_=self.get_reachable_point()
                    start_goal_distance=np.linalg.norm(self.goal_1 - self.goal_2)
                    if start_goal_distance<self.min_distance:
                        points_found=False
                    else:
                        alpha_min=self.min_distance/start_goal_distance
                        alpha_max=1-alpha_min
                        alpha =  np.random.uniform(alpha_min, alpha_max)
                        self.avoid_2=(1-alpha)*self.goal_1+alpha*self.goal_2
                        avoid_2_is_reachable,_=self.is_reachable(self.avoid_2)
                        if avoid_2_is_reachable:
                            points_found=True
                        else:
                            points_found=False
                else:
                    points_found=False

        if self.video_mode:
            self.set_and_move_graphic_balls()
        
        self.robot.set(self.initial_pose)

        self.setting={'initial_pose':self.initial_pose,'goal_1':self.goal_1,'avoid_1':self.avoid_1,'goal_2':self.goal_1,'avoid_2':self.avoid_2}

        with open(os.path.join(self.output_path,'setting.pkl'),'wb') as f:
            pickle.dump(self.setting,f)

    
    def reset(self,**kwargs):

        self.first_complete=False
        self.goal_index=0
        
        self.steps=0
        
        self.robot.set(self.initial_pose)

        self.requirement_evaluator.reset_signals()
        #self.reward_evaluator.reset_signals()
        self.first_part_evaluator.reset_signals()  
        self.second_part_evaluator.reset_signals()
        #self.first_part_hard_evaluator.reset_signals()
        #self.second_part_hard_evaluator.reset_signals()
        
        self.requirement_evaluating_function=self.requirement_evaluator.apply_formula()
        #self.reward_evaluating_function=self.reward_evaluator.apply_formula()
        self.first_part_evaluating_function=self.first_part_evaluator.apply_formula()
        self.second_part_evaluating_function=self.second_part_evaluator.apply_formula()
        #self.first_part_hard_evaluating_function=self.first_part_hard_evaluator.apply_formula()
        #self.second_part_hard_evaluating_function=self.second_part_hard_evaluator.apply_formula()


        observation=self._get_obs()
        reset_info={} #needed for stable baseline

        if self.video_mode:
            self.frames=[]

        return observation,reset_info
    
    def set_and_move_graphic_balls(self):
        self.goal_ball_1.changePos(self.goal_1, 4)
        self.avoid_ball_1.changePos(self.avoid_1, 4)
        self.goal_ball_2.changePos(self.goal_2, 4)
        self.avoid_ball_2.changePos(self.avoid_2, 4)

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
        truncated=False

        distance_from_goal_1=self.distance_from_goal_1()
        distance_from_goal_2=self.distance_from_avoid_1()
        distance_from_avoid_1=self.distance_from_avoid_1()
        distance_from_avoid_2=self.distance_from_avoid_2()

        goal_signal_1=self.sphere_radius-distance_from_goal_1
        avoid_collision_signal_1=distance_from_avoid_1-self.sphere_radius
        avoid_near_signal_1=1.5*(distance_from_avoid_1-self.hard_sphere_radius)

        goal_signal_2=self.sphere_radius-distance_from_goal_2
        avoid_collision_signal_2=distance_from_avoid_2-self.sphere_radius
        avoid_near_signal_2=1.5*(distance_from_avoid_2-self.hard_sphere_radius)

        signals=np.array([goal_signal_1,avoid_collision_signal_1,avoid_near_signal_1,goal_signal_2,avoid_collision_signal_2,avoid_near_signal_2])

        self.requirement_evaluator.append_signals(signals)
        #self.reward_evaluator.append_signals(signals)
        self.first_part_evaluator.append_signals(signals)
        self.second_part_evaluator.append_signals(signals)
        #self.first_part_hard_evaluator.append_signals(signals)
        #self.second_part_hard_evaluator.append_signals(signals)


        if self.goal_index==0:
            reward=self.first_part_evaluating_function(0)
            if reward>0:
            #if self.first_part_evaluating_function(0)>0:
                self.first_complete=True
            #if self.steps>self.max_steps/2:
                self.goal_index+=1
                #self.robot.set(self.second_initial_pose)
                #self.reach_first_step=self.steps
        else:
            reward=self.second_part_evaluating_function(0)
            if reward>0:
                print("reached both")
        
        info={'episode_number':self.episodes,'step':self.steps,'requirement_robustness':0}

        if self.steps>self.max_steps:
            robustness=self.requirement_evaluating_function(0)
            terminated=True
            if robustness>0 or (self.first_complete and reward>0):
                end_condition='perfect'
            elif self.first_complete:
                end_condition='first_part_completed_but_not_second'
            else:
                end_condition='no_part_completed'

            info['end_condition']=end_condition
            info['requirement_robustness']=self.requirement_evaluating_function(0)
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
        self.goal_ball_1=Ball(self.urdf_dir,color="green")
        self.avoid_ball_1=Ball(self.urdf_dir,color="red")
        self.goal_ball_2=Ball(self.urdf_dir,color="green")
        self.avoid_ball_2=Ball(self.urdf_dir,color="red")
        self.video_mode=True

    def disable_video_mode(self):
        self.video_mode=False

    def _get_obs(self):
        observation=self.robot.observe()[:6]
        position_of_end_effector=self.get_position_of_end_effector()
        obs=np.concatenate([observation,position_of_end_effector,[self.goal_index]])
        return obs
    
    def close(self):
        self.robot.close()
    def distance_from_goal_1(self):
        return np.linalg.norm(self.goal_1-self.get_position_of_end_effector())
    def distance_from_goal_2(self):
        return np.linalg.norm(self.goal_2-self.get_position_of_end_effector())
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
    
