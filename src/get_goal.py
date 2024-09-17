from gym_ergojr.utils.math import RandomPointInHalfSphere
from gym_ergojr.sim.single_robot import SingleRobot
import numpy as np
import pybullet as p


rhis = RandomPointInHalfSphere(0.0,0.0369,0.0437,radius=0.2022,height=0.2610,min_dist=0.1)

"""
for i in range(2):
    point=rhis.samplePoint()
    print(point)
"""

robot = SingleRobot()

for i in range(600):
    action=np.random.beta(0.5, 0.5, 6) * 2 - 1
    num_steps = np.random.randint(10, 50)  # Randomize number of consecutive steps

    for _ in range(num_steps):
        robot.act2(action)
        robot.step()
    if i==200 or i==400 or i==599:
        print(np.array(p.getLinkState(robot.id,13)[0]))
