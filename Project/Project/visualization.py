import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define file path
file_path = "Project/Project/dataset/dataset.hdf5"

# Specify the dataset to visualize (change this to the specific file name)
dataset_name = "Charlotte_Walking"  # Update this as needed

with h5py.File(file_path, "r") as f:
    if "raw" in f and dataset_name in f["pre-processed"]:
        data = f["pre-processed"][dataset_name][:]
    else:
        raise FileNotFoundError(f"Dataset '{dataset_name}' not found in 'raw' group.")

# Convert to DataFrame
columns = ["Time", "Linear Acceleration x (m/s^2)", "Linear Acceleration y (m/s^2)", 
           "Linear Acceleration z (m/s^2)", "Absolute acceleration (m/s^2)"]
df = pd.DataFrame(data, columns=columns)

# Skip first row (assuming labels)
time = df.iloc[1:, 0]
acceleration = df.iloc[1:, 1:]

# Choose every 500th element
av_time = time.iloc[::500]
av_acceleration = acceleration.iloc[::500, :]

# Visualization
acceleration_columns = columns[1:]
num_plots = len(acceleration_columns)
fig, ax = plt.subplots(ncols=1, nrows=num_plots, figsize=(5 * num_plots, 5), sharex=True)

for i, col in enumerate(acceleration_columns):
    ax[i].plot(av_time, av_acceleration[col], label=col, color='b', marker='o', linestyle='-')
    ax[i].set_xlabel("Time (s)")
    ax[i].set_ylabel(col)
    ax[i].set_title(f"{col} vs Time")
    ax[i].grid(True)
    ax[i].legend()

fig.tight_layout()
plt.show()
