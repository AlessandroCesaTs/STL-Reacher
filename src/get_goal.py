from gym_ergojr.utils.math import RandomPointInHalfSphere

rhis = RandomPointInHalfSphere(0.0,0.0369,0.0437,radius=0.2022,height=0.2610,min_dist=0.1)

point=rhis.samplePoint()
print(point)