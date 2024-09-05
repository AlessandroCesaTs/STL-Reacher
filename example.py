import numpy as np

from simplemonitor.stl import STLMonitorBuilder, RobSemantics, TimeSeries

timeSeries = TimeSeries(['P', 'Q'],  # variables
                        np.array([0, 1, 2, 3, 4]),  # timesteps
                        np.array([[5, 67, -6, -1, -1],  # values assumed by the first variable for each timestep
                                  [3, 3, -3, -1, -1]]))  # values assumed by the first variable for each timestep

# formula = '(G_[1,6] (Q>2)) | (!(F_[1,10](P<=4)))'
formula = '(G_[1,6] (Q>-20))'  # STL formula
monitor = STLMonitorBuilder(formula).build()
robustness_value = monitor.monitor(RobSemantics(timeSeries=timeSeries, currentState=0))
print(robustness_value)
print("DONE")