import os
import numpy as np
import gymnasium as gym
import pybullet as p
import cv2
from gymnasium import spaces
from gym_ergojr.sim.single_robot import SingleRobot
from gym_ergojr.sim.objects import Ball
from gym_ergojr.utils.pybullet import DistanceBetweenObjects

class MyReacherEnv(gym.Env):
    def __init__(self,video_path):
        super().__init__()
        self.max_force=1
        self.max_vel=18
        self.robot = SingleRobot()
        self.ball = Ball()
        self.observation_space = spaces.Box(low=-1, high=1, shape=(12,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32) 
        self.steps=0
        self.max_steps=1000
        self.goal=np.array([-0.012135819251746286, -0.0843113137763625, 0.16126595580699604])
        self.avoid=np.array([-0.04458391394299169, -0.009719424686773021, 0.20094343703790016])
        self.avoid_ball=Ball()
        self.dist = DistanceBetweenObjects(
            bodyA=self.robot.id, bodyB=self.ball.id, linkA=13, linkB=1)
        self.avoid_dist = DistanceBetweenObjects(
            bodyA=self.robot.id, bodyB=self.avoid_ball.id, linkA=13, linkB=1)
        self.GOAL_REACHED_DISTANCE = -0.016  # distance between robot tip and goal under which the task is considered solved
        

        self.episodes=0

        self.video_mode=False
        self.frames=[]

        os.makedirs(video_path, exist_ok=True)
        self.video_path=os.path.join(video_path,'simulation.avi')
        self.image_size=(640,480)
        self.fps=10

    def reset(self,**kwargs):
        self.episodes+=1
        self.steps=0
        
        qpos=np.zeros(12)
        self.robot.reset()
        self.robot.set(qpos)
        self.robot.act2(qpos[:6])
        self.robot.step()

        self.dist.goal = self.goal
        self.ball.changePos(self.goal, 4)
        self.avoid_dist.goal = self.avoid
        self.avoid_ball.changePos(self.avoid, 4)
        for _ in range(25):
            self.robot.step()

        observation=self._get_obs()
        reset_info={} #needed for stable baseline

        if self.video_mode:
            self.frames=[]

        return observation,reset_info

    def _getReward(self):
        terminated=False
        truncated = False

        reward = self.dist.query()
        distance = reward.copy()

        distance_from_avoid=self.avoid_dist.query()
        
        reward *= -1  # the reward is the inverse distance
        if distance_from_avoid<-self.GOAL_REACHED_DISTANCE:
            truncated=True
            reward=-1
        elif reward > self.GOAL_REACHED_DISTANCE:  # this is a bit arbitrary, but works well
            terminated = True
            reward = 1

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

        self.robot.act2(action, max_force=self.max_force, max_vel=self.max_vel)
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

