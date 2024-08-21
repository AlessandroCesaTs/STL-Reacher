from MyReacherEnv import MyReacherEnv
import time

env=MyReacherEnv(visual=True)

env.reset()
print("Reset")
time.sleep(5)

print("start simulating")
for i in range(60):
    print(i)
    time.sleep(0.5)
    obs, reward, terminated, truncated, info=env.step([0.26954633, -0.82377454, 0.18293788, -0.56170284, 0.10516296,
    0.79996014])
    if i%20==0 and i!=0:
        env.reset()
        print("Reset")
        time.sleep(5)
    

