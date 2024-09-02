import pandas as pd
import matplotlib.pyplot as plt

# Read the data from the CSV file
df = pd.read_csv('output_epyc/times.csv')

# Calculate the mean time for each number of cores
mean_times = df.groupby("Cores").median()

# Plotting the mean times
plt.figure(figsize=(8, 6))
plt.plot(mean_times['Time'], marker='o')
plt.title('Mean Times for Each Number of Cores')
plt.xlabel('Number of Cores')
plt.ylabel('Mean Time (seconds)')
plt.grid(True)
plt.savefig('epyc_scaling.png')
