import pandas as pd
import matplotlib.pyplot as plt

# Read the data from the CSV file
df = pd.read_csv('output_thin/times.csv')

# Calculate the mean time for each number of cores
median_times = df.groupby("Cores").median()

mean_times = df.groupby("Cores").mean()

# Plotting the mean times
plt.figure(figsize=(8, 6))
plt.plot(mean_times['Time'], marker='o')
plt.title('Mean Times for Each Number of Cores')
plt.xlabel('Number of Cores')
plt.ylabel('Mean Time (seconds)')
plt.grid(True)
plt.savefig('thin_mean_scaling.png')


# Plotting the mean times
plt.figure(figsize=(8, 6))
plt.plot(median_times['Time'], marker='o')
plt.title('Median Times for Each Number of Cores')
plt.xlabel('Number of Cores')
plt.ylabel('Median Time')
plt.grid(True)
plt.savefig('thin_median_scaling.png')
# Read the data from the CSV file


df = pd.read_csv('output_epyc/times.csv')

# Calculate the mean time for each number of cores
median_times = df.groupby("Cores").median()

mean_times = df.groupby("Cores").mean()

# Plotting the mean times
plt.figure(figsize=(8, 6))
plt.plot(mean_times['Time'], marker='o')
plt.title('Mean Times for Each Number of Cores')
plt.xlabel('Number of Cores')
plt.ylabel('Mean Time (seconds)')
plt.grid(True)
plt.savefig('epyc_mean_scaling.png')


# Plotting the mean times
plt.figure(figsize=(8, 6))
plt.plot(median_times['Time'], marker='o')
plt.title('Median Times for Each Number of Cores')
plt.xlabel('Number of Cores')
plt.ylabel('Median Time')
plt.grid(True)
plt.savefig('epyc_median_scaling.png')

