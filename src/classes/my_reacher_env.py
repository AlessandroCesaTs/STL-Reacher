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

from utils.utils import copy_urdf_directory

urdf_default_dir='env/lib/python3.12/site-packages/gym_ergojr/scenes/'

class MyReacherEnv(gym.Env):
    def __init__(self,urdf_dir=urdf_default_dir,max_steps=100,visual=False,output_path=os.getcwd()):
        super().__init__()
        self.observation_space = spaces.Box(low=-1, high=1, shape=(19,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(5,), dtype=np.float32)
        self.output_path=output_path

        self.video_mode=False

        self.urdf_dir=copy_urdf_directory(urdf_dir)
        self.rhis = RandomPointInHalfSphere(0,0,0,radius=0.2,min_dist=0.1)

        self.robot = SingleRobot(debug=visual,urdf_dir=self.urdf_dir)

        self.min_distance=0.1

        self.sphere_radius = 0.02  # distance between robot tip and goal under which the task is considered solved
        self.min_distances=0.1
        self.soft_distance=0.05
        self.max_iterations_to_check_point=500

        self.steps=0
        self.start_computing_robustness_from=0
        self.max_steps=max_steps  
        self.episodes=0

        signals=[[],[],[]]
        
        self.soft_requirement=["and",["F",0],["G",1]]
        self.hard_requirement=["G",2]

        self.soft_evaluator=STLEvaluator(signals,self.soft_requirement)
        self.hard_evaluator=STLEvaluator(signals,self.hard_requirement)

        self.soft_function=self.soft_evaluator.apply_formula()
        self.hard_function=self.hard_evaluator.apply_formula()

    def new_start_goal_avoid(self):

        self.starting_point,initial_pose=self.get_reachable_point()
        
        points_found=False
        while not points_found:
            self.goal,_=self.get_reachable_point()
            start_goal_distance=np.linalg.norm(self.starting_point - self.goal)
            if start_goal_distance<self.min_distances:
                points_found=False
            else:
                alpha_min=self.min_distances/start_goal_distance
                alpha_max=1-alpha_min
                alpha =  np.random.uniform(alpha_min, alpha_max)
                self.avoid=(1-alpha)*self.starting_point+alpha*self.goal
                avoid_is_reachable,_=self.is_reachable(self.avoid)
                if avoid_is_reachable:
                    points_found=True
                else:
                    points_found=False

        self.flatten_points=np.concatenate([self.starting_point,self.goal,self.avoid])

        if self.video_mode:
            self.goal_balls=Ball(self.urdf_dir,color="green")
            self.avoid_balls=Ball(self.urdf_dir,color="red")
            
            self.set_and_move_graphic_balls()
        
        self.robot.set(initial_pose)

    
    def reset(self,**kwargs):
        self.steps=0
        
        self.new_start_goal_avoid()

        self.soft_evaluator.reset_signals()
        self.hard_evaluator.reset_signals()

        self.soft_function=self.soft_evaluator.apply_formula()
        self.hard_function=self.hard_evaluator.apply_formula()

        observation=self._get_obs()
        reset_info={} #needed for stable baseline

        if self.video_mode:
            self.frames=[]

        return observation,reset_info
    
    def set_and_move_graphic_balls(self):
        for i in range(self.num_of_goals):
            self.goal_balls[i].changePos(self.goals[i], 4)
        for i in range(self.num_of_avoids):
            self.avoid_balls[i].changePos(self.avoids[i], 4)

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
        avoid_soft_signal=10*(distance_from_avoid-self.soft_distance)
        avoid_hard_signal=distance_from_avoid-self.sphere_radius

        signals=np.array([goal_signal,avoid_soft_signal,avoid_hard_signal])
        self.soft_evaluator.append_signals(signals)
        self.hard_evaluator.append_signals(signals)

        reward=self.soft_function(0)
        safety=self.hard_function(0)
        
        info={'episode_number':self.episodes,'step':self.steps,'safety':safety,'distances':distance_from_goal}            
        
        if reward>0:
            terminated=True
            if safety>self.soft_distance-self.sphere_radius:
                end_condition='perfect'
            else:
                end_condition='danger'
        #elif safety<=0 :
        #    truncated=True
        #    end_condition='collision'
        elif self.steps>self.max_steps:
            truncated=True
            end_condition='too_many_steps'
        
        if terminated or truncated:
            if safety<0:
                end_condition='collision'
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
        self.video_mode=True

    def disable_video_mode(self):
        self.video_mode=False

    def _get_obs(self):
        observation=self.robot.observe()
        obs=np.concatenate([observation,self.flatten_points])
        return obs
    
    def close(self):
        self.robot.close()
    def distance_from_goal(self):
        return np.linalg.norm(self.goal-self.get_position_of_end_effector())
    def distance_from_avoid(self):
        return np.linalg.norm(self.avoid-self.get_position_of_end_effector())
    def get_position_of_end_effector(self):
        return np.array(p.getLinkState(self.robot.id,13)[0])

    def get_reachable_point(self):
        reachable=False
        while not reachable:
            point=self.rhis.samplePoint()
            reachable, joints_positions_velocities = self.is_reachable(point)
        self.robot.set(np.zeros(10))
        return point,joints_positions_velocities

    def is_reachable(self, point):
        distance=np.inf
        iter=0
        while distance>self.sphere_radius and iter<self.max_iterations_to_check_point:
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
    
        

