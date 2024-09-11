import os
import numpy as np
import gymnasium as gym
import pybullet as p
import cv2
import time
from gymnasium import spaces
from gym_ergojr.sim.single_robot import SingleRobot
from gym_ergojr.sim.objects import Ball
from gym_ergojr.utils.math import RandomPointInHalfSphere
from classes.stl_evaluator import STLEvaluator


from utils.utils import copy_urdf_directory

urdf_default_dir='env/lib/python3.12/site-packages/gym_ergojr/scenes/'

class MyReacherEnv(gym.Env):
    def __init__(self,urdf_dir=urdf_default_dir,num_of_goals=1,num_of_avoids=1,max_steps=100,visual=False,output_path=os.getcwd()):
        super().__init__()
        self.observation_space = spaces.Box(low=-1, high=1, shape=(12+num_of_goals*3+num_of_avoids*3,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32)
        self.output_path=output_path

        self.num_of_goals=num_of_goals
        self.num_of_avoids=num_of_avoids

        self.urdf_dir=copy_urdf_directory(urdf_dir)
        self.rhis = RandomPointInHalfSphere(0.0,0.0369,0.0437,radius=0.2022,height=0.2610,min_dist=0.1)


        self.goals=np.array([[0.03325928, 0.05016471, 0.16669361],[ 0.15904594, -0.00376019,  0.13412063]])
        self.avoids=np.array([])
        normalized_goals=np.array([self.rhis.normalize(goal) for goal in self.goals])
        #normalized_avoids=np.array([self.rhis.normalize(avoid) for avoid in self.avoids])
        #self.flatten_goals_and_avoids=np.concatenate([normalized_goals.flatten(),normalized_avoids.flatten()])
        self.flatten_goals_and_avoids=normalized_goals.flatten()
        self.goal_to_reach=0

        self.robot = SingleRobot(debug=visual,urdf_dir=self.urdf_dir)
        self.goal_balls=[Ball(self.urdf_dir,color="green"),Ball(self.urdf_dir,color="blue")]
        self.avoid_balls=[Ball(self.urdf_dir,color="red") for _ in range(self.num_of_avoids)]
        
        self.set_and_move_graphic_balls()
        
        self.min_distance=0.1
        self.rhis = RandomPointInHalfSphere(0.0,0.0369,0.0437,radius=0.2022,height=0.2610,min_dist=0.1)

        self.goal_sphere_radius = 0.02  # distance between robot tip and goal under which the task is considered solved

        self.steps=0
        self.max_steps=max_steps  
        self.goal_to_reach_index=0
        self.episodes=0

        self.video_mode=False
        self.frames=[]
        os.makedirs(os.path.join(output_path,'videos'), exist_ok=True)
        self.video_path=os.path.join(output_path,'videos','simulation.avi')
        self.image_size=(640,480)
        self.fps=5

        signals=[[] for _ in range (self.num_of_goals+self.num_of_avoids)]

        self.stl_formulas=[["F",0],["F",1],["F",["and",0,["F",1]]]]
        self.stl_evaluators=[]
        self.stl_formula_evaluators=[]
        for i in range(len(self.stl_formulas)):
            self.stl_evaluators.append(STLEvaluator(signals,self.stl_formulas[i]) )
            self.stl_formula_evaluators.append(self.stl_evaluators[i].apply_formula())


    def reset(self,**kwargs):
        self.episodes+=1
        self.steps=0
        
        self.robot.reset()

        for i in range(len(self.stl_formulas)):
            self.stl_evaluators[i].reset_signals()
            self.stl_formula_evaluators[i]=self.stl_evaluators[i].apply_formula()

        #self.set_goals_and_avoids()

        observation=self._get_obs()
        reset_info={} #needed for stable baseline

        if self.video_mode:
            self.frames=[]

        self.goal_to_reach_index=0

        return observation,reset_info

    def set_and_move_graphic_balls(self):
        for i in range(self.num_of_goals):
            self.goal_balls[i].changePos(self.goals[i], 4)
        for i in range(self.num_of_avoids):
            self.avoid_balls[i].changePos(self.avoids[i], 4)

        for _ in range(25):
            self.robot.step()

    def _getReward(self):
        terminated=False
        truncated = False

        distances_from_goals=self.distances_from_goals()
        robustnesses=np.zeros(len(self.stl_formulas))
        for i in range(len(self.stl_formulas)):
            for j in range(self.num_of_goals):
                self.stl_evaluators[i].append_single_signal(j,self.goal_sphere_radius-distances_from_goals[j])
            robustnesses[i]=self.stl_formula_evaluators[i](0)

        reward=robustnesses[self.goal_to_reach]
        #print(f"print robustness are {robustnesses} goal to reach is {self.goal_to_reach} reward is {reward}")

        if reward>0:
            if self.goal_to_reach<self.num_of_goals-1:
                self.goal_to_reach+=1
            else:
                terminated=True
        elif self.steps>self.max_steps:
            truncated=True
        
        info={'robustnesses':robustnesses}

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

    def save_video(self):
        if self.frames:
            out=cv2.VideoWriter(self.video_path,cv2.VideoWriter_fourcc(*'XVID'),fps=self.fps,frameSize=self.image_size)
            for image in self.frames:
                out.write(image)
            out.release()
        

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

    def enable_video_mode(self):
        self.video_mode=True

    def disable_video_mode(self):
        self.video_mode=False
        self.frames=[]

    def _get_obs(self):
        obs = np.concatenate([self.robot.observe(),self.flatten_goals_and_avoids])
        return obs
    
    def close(self):
        self.robot.close()
    def distances_from_goals(self):
        return np.linalg.norm(self.goals-self.get_position_of_end_effector(),axis=1)
    def distance_from_avoid(self):
        return np.array(np.linalg.norm(self.avoids-self.get_position_of_end_effector(),axis=1))
    def get_position_of_end_effector(self):
        return np.array(p.getLinkState(self.robot.id,13)[0])

    def set_goals_and_avoids(self):
        goals=[]
        avoids=[]

        while len(goals)<self.num_of_goals:
            goal=self.rhis.samplePoint()
            #goal=self.get_reachable_position()
            if self.is_valid_position(goal,avoids):
                goals.append(goal)
        
        while len(avoids)<self.num_of_avoids:
            avoid=self.rhis.samplePoint()
            #avoid=self.get_reachable_position()
            if self.is_valid_position(avoid,goals):
                avoids.append(avoid)
        self.goals=np.array(goals[:self.num_of_goals])
        self.avoids=np.array(avoids[:self.num_of_avoids])
        normalized_goals=np.array([self.rhis.normalize(goal) for goal in self.goals])
        normalized_avoids=np.array([self.rhis.normalize(avoid) for avoid in self.avoids])
        self.flatten_goals_and_avoids=np.concatenate([normalized_goals.flatten(),normalized_avoids.flatten()])

        self.set_and_move_graphic_balls()

    def is_valid_position(self,point,other_points):
        if np.linalg.norm(point-self.get_position_of_end_effector())<self.min_distance:
            return False
        else:
            for other_point in other_points:
                if np.linalg.norm(point-other_point)<self.min_distance:
                    return False
        return True

    def get_reachable_position(self):
        for _ in range(50):
            action=np.random.beta(0.5, 0.5, 6) * 2 - 1
            num_steps = np.random.randint(5, 15)  # Randomize number of consecutive steps

            for _ in range(num_steps):
                self.robot.act2(action)
                self.robot.step()
        pos=self.get_position_of_end_effector()
        self.robot.reset()
        
        return pos
  

    
