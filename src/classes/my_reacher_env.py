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
        self.observation_space = spaces.Box(low=-1, high=1, shape=(21,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32)
        self.output_path=output_path

        self.video_mode=False

        self.urdf_dir=copy_urdf_directory(urdf_dir)
        self.rhis = RandomPointInHalfSphere(0.0,0.0369,0.0437,radius=0.2022,height=0.2610,min_dist=0.1)

        self.robot = SingleRobot(debug=visual,urdf_dir=self.urdf_dir)

        self.min_distance=0.1

        self.goal_sphere_radius = 0.02  # distance between robot tip and goal under which the task is considered solved

        self.steps=0
        self.start_computing_robustness_from=0
        self.max_steps=max_steps  
        self.episodes=0

        signals=[[],[]]
        
        self.stl_formula=["and",["F",0],["G",1]]
        self.reward_evaluator=STLEvaluator(signals,self.stl_formula)
        self.reward_formula_evaluator=self.reward_evaluator.apply_formula()

        self.safety_evaluator=STLEvaluator(signals,self.stl_formula[-1])
        self.safety_formula_evaluator=self.safety_evaluator.apply_formula()

    def new_start_goal_avoid(self):

        self.robot_initial_pose=np.concatenate((np.random.uniform(-1,1,6),np.zeros(6)))
        self.robot.set(self.robot_initial_pose)
        self.starting_point=self.get_position_of_end_effector()
        self.goal=self.rhis.samplePoint()
        alpha=alpha = np.random.rand()
        self.avoid=(1-alpha)*self.starting_point+alpha*self.goal
        normalized_starting_point=self.rhis.normalize(self.starting_point)
        normalized_goal=self.rhis.normalize(self.goal)
        normalized_avoid=self.rhis.normalize(self.avoid)
        self.flatten_points=np.concatenate([normalized_starting_point.flatten(),normalized_goal.flatten(),normalized_avoid.flatten()])

        if self.video_mode:
            self.goal_balls=Ball(self.urdf_dir,color="green")
            self.avoid_balls=Ball(self.urdf_dir,color="red")
            
            self.set_and_move_graphic_balls()

    
    def reset(self,**kwargs):
        self.steps=0
        
        self.new_start_goal_avoid()

        self.reward_evaluator.reset_signals()
        self.reward_formula_evaluator=self.reward_evaluator.apply_formula()

        self.safety_evaluator.reset_signals()
        self.safety_formula_evaluator=self.safety_evaluator.apply_formula()

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
        goal_signal=self.goal_sphere_radius-distance_from_goal
        avoid_signal=distance_from_avoid-self.goal_sphere_radius
        signals=np.array([goal_signal,avoid_signal])
        self.reward_evaluator.append_signals(signals)
        self.safety_evaluator.append_signals(signals)

        reward=self.reward_formula_evaluator(0)
        safety=self.safety_formula_evaluator(0)

        if reward>0:
            terminated=True
        elif (safety<0) or self.steps>self.max_steps:
            truncated=True
        
        info={'episode_number':self.episodes,'step':self.steps,'safety':safety,'distances':distance_from_goal}            
        
        if terminated or truncated:
            final_boolean=int(reward>0)
            info['final_boolean']=final_boolean
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

