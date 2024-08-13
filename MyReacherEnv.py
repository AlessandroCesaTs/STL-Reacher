import numpy as np
import pybullet as p
from gym_ergojr.envs.ergo_reacher_env import ErgoReacherEnv

class MyReacherEnv(ErgoReacherEnv):
    def __init__(self,env):
        super().__init__(env,simple=True)
        self.GOAL_REACHED_DISTANCE = -0.016  # distance between robot tip and goal under which the task is considered solved
        self.RADIUS = 0.2022
        self.DIA = 2 * self.RADIUS
        self.RESET_EVERY = 5  # for the gripper

    def reset(self,**kwargs):
        self.goals_done=0
        self.episodes+=1
        
        if self.episodes >= self.restart_every_n_episodes:
            self.robot.hard_reset()  # this always has to go first
            self.ball.hard_reset()
            self._setDist()
            self.episodes = 0

        qpos=np.zeros(12)
        self.robot.reset()
        self.robot.set(qpos)
        self.robot.act2(qpos[:6])
        self.robot.step()

        self.goal=np.array([0,0,0])
        self.dist.goal = self.goal
        self.ball.changePos(self.goal, 4)

        self.is_initialized = True

        observation=self._get_obs()
        reset_info={} #needed for stable baseline

        return observation,reset_info





    def step(self, action):
        obs,reward,done,info=super().step(action)
        terminated=done
        truncated=False
        return obs,reward,terminated,truncated,info