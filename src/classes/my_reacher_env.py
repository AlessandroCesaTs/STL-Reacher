import os
import numpy as np
import gymnasium as gym
import pybullet as p
import cv2
from gymnasium import spaces
from gym_ergojr.sim.single_robot import SingleRobot
from gym_ergojr.sim.objects import Ball

class MyReacherEnv(gym.Env):
    def __init__(self,num_of_goals=1,num_of_avoids=1,max_steps=1024,steps_to_set_goal=200,visual=False,output_path=os.getcwd()):
        super().__init__()
        self.observation_space = spaces.Box(low=-1, high=1, shape=(12,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32)

        self.num_of_goals=num_of_goals
        self.num_of_avoids=num_of_avoids
        self.steps_to_set_goal=steps_to_set_goal

        self.robot = SingleRobot(debug=visual)
        self.goal_balls=[Ball(color="green") for _ in range(self.num_of_goals)]
        self.avoid_balls=[Ball(color="red") for _ in range(self.num_of_avoids)]
        
        self.min_distance_between_goal_and_avoid=0.02
        self.set_goals_and_avoids()

        for i in range(self.num_of_goals):
            self.goal_balls[i].changePos(self.goals[i], 4)
        for i in range(self.num_of_avoids):
            self.avoid_balls[i].changePos(self.avoids[i], 4)

        for _ in range(25):
            self.robot.step()

        self.goal_sphere_radius = -0.016  # distance between robot tip and goal under which the task is considered solved

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

    def reset(self,**kwargs):
        self.episodes+=1
        self.steps=0
        
        self.robot.reset()

        observation=self._get_obs()
        reset_info={} #needed for stable baseline

        if self.video_mode:
            self.frames=[]

        self.goal_to_reach_index=0

        return observation,reset_info

    def _getReward(self):
        terminated=False
        truncated = False

        reward=self.distance_from_goal()
        distance = reward.copy()

        distance_from_avoid=self.distance_from_avoid()
        
        reward *= -1  # the reward is the inverse distance
        if distance_from_avoid<-self.goal_sphere_radius:
            truncated=True
            reward=-1
        elif reward > self.goal_sphere_radius:  # this is a bit arbitrary, but works well
            reward = 1
            if self.goal_to_reach_index <self.num_of_goals-1:
                self.goal_to_reach_index+=1
            else:
                terminated = True                

        return reward, terminated, truncated, distance

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

        reward, terminated, truncated, dist = self._getReward()

        obs = self._get_obs()

        self.steps+=1

        if not truncated:
            truncated=self.steps>self.max_steps

        if self.video_mode:
            image=self._capture_image()
            self.frames.append(image)
        
        return obs,reward,terminated,truncated,{"distance": dist}

    def enable_video_mode(self):
        self.video_mode=True

    def disable_video_mode(self):
        self.video_mode=False
        self.frames=[]

    def _get_obs(self):
        obs = self.robot.observe()
        return obs
    
    def close(self):
        self.robot.close()
    def distance_from_goal(self):
        goal_to_reach=self.goals[self.goal_to_reach_index]
        return np.linalg.norm(goal_to_reach-self.get_position_of_end_effector())
    def distance_from_avoid(self):
        return np.linalg.norm(self.avoids-self.get_position_of_end_effector())
    def get_position_of_end_effector(self):
        return np.array(p.getLinkState(self.robot.id,13)[0])

    def set_goals_and_avoids(self):
        goals=[]
        avoids=[]

        while len(goals)<self.num_of_goals or len(avoids)<self.num_of_avoids:
            goal,avoid=self.get_goal()
            if self.is_valid_position(goal,avoids):
                goals.append(goal)
            if self.is_valid_position(avoid,goals):
                avoids.append(avoid)

        self.goals=np.array(goals[:self.num_of_goals])
        self.avoids=np.array(avoids[:self.num_of_avoids])

    def get_goal(self):
        avoid=None
        for i in range(self.steps_to_set_goal):
            if i%(self.steps_to_set_goal//4)==0:
                action=np.random.uniform(-1,1,6)
            self.robot.act2(action)
            self.robot.step()
            if i%(self.steps_to_set_goal//2)==0:
                avoid=self.get_position_of_end_effector()
        goal=self.get_position_of_end_effector()
        return goal, avoid
    
    def is_valid_position(self,point,other_points):
        for other_point in other_points:
            if np.linalg.norm(point-other_point)<self.min_distance_between_goal_and_avoid:
                return False
        return True

