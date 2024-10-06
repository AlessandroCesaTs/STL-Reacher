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

class MyReacherEnv(gym.Env):
    def __init__(self,urdf_dir=urdf_default_dir,max_steps=100,visual=False,output_path=os.getcwd(),change_target=False):
        super().__init__()
        self.observation_space = spaces.Box(low=-1, high=1, shape=(19,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(5,), dtype=np.float32)
        self.output_path=output_path

        self.video_mode=False
        self.change_target=change_target

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

        signals=[[],[]]
        

        self.reach_formula=["F",0]
        self.stay_formula=["F",["G",0]]
        self.collision_formula=["G",1]
        self.requirement=["and",self.stay_formula,self.collision_formula]

        self.evaluator=STLEvaluator(signals,self.requirement)
        self.reach_evaluator=STLEvaluator(signals,self.reach_formula)
        self.stay_evaluator=STLEvaluator(signals,self.stay_formula)
        self.collision_evaluator=STLEvaluator(signals,self.collision_formula)

        self.evaluating_function=self.evaluator.apply_formula()
        self.reach_evaluating_function=self.reach_evaluator.apply_formula()
        self.stay_evaluating_function=self.stay_evaluator.apply_formula()
        self.collision_evaluating_function=self.collision_evaluator.apply_formula()

        if not self.change_target:
            self.new_start_goal_avoid()

    def set_start_goal_avoid_from_file(self):
        with open(os.path.join(self.output_path,'setting.pkl'), 'rb') as f:
            setting = pickle.load(f)
        self.starting_point=setting['starting_point']
        initial_pose=setting['initial_pose']
        self.goal=setting['goal']
        self.avoid=setting['avoid']
        self.flatten_points=np.concatenate([self.goal,self.avoid])
        self.robot.set(initial_pose)

        if self.video_mode:
            self.set_and_move_graphic_balls()


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

        self.flatten_points=np.concatenate([self.goal,self.avoid])

        if self.video_mode:
            self.set_and_move_graphic_balls()
        
        self.robot.set(initial_pose)
        if not self.change_target:
            setting={'starting_point':self.starting_point, 'initial_pose':initial_pose,'goal':self.goal,'avoid':self.avoid}
            with open(os.path.join(self.output_path,'setting.pkl'),'wb') as f:
                pickle.dump(setting,f)

    
    def reset(self,**kwargs):
        self.steps=0
        
        if self.change_target:
            self.new_start_goal_avoid()

        self.evaluator.reset_signals()
        self.reach_evaluator.reset_signals()
        self.stay_evaluator.reset_signals()
        self.collision_evaluator.reset_signals()

        self.evaluating_function=self.evaluator.apply_formula()
        self.reach_evaluating_function=self.reach_evaluator.apply_formula()
        self.stay_evaluating_function=self.stay_evaluator.apply_formula()
        self.collision_evaluating_function=self.collision_evaluator.apply_formula()

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
        avoid_signal=distance_from_avoid-self.sphere_radius

        signals=np.array([goal_signal,avoid_signal])
        self.evaluator.append_signals(signals)
        self.reach_evaluator.append_signals(signals)
        self.stay_evaluator.append_signals(signals)
        self.collision_evaluator.append_signals(signals)

        reward=self.evaluating_function(0)

        
        info={'episode_number':self.episodes,'step':self.steps,'distances':distance_from_goal}            
        
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
        self.goal_balls=Ball(self.urdf_dir,color="green")
        self.avoid_balls=Ball(self.urdf_dir,color="red")
        self.video_mode=True

    def disable_video_mode(self):
        self.video_mode=False

    def _get_obs(self):
        observation=self.robot.observe()
        position_of_end_effector=self.get_position_of_end_effector()
        obs=np.concatenate([observation,position_of_end_effector,self.flatten_points])
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
    
        

