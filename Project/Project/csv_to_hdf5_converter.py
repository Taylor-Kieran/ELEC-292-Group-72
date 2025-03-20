import os
import pandas as pd
from pandasgui import show
import h5py
import numpy as np
import matplotlib.pyplot as plt

# Define paths
raw_data_folder = "C:\\Users\\charl\\.vscode\\290\\ELEC-292-Group-72\\Project\\Project"  # Folder containing raw CSV files
dataset_folder = "dataset"  # Folder where HDF5 will be stored
hdf5_path = os.path.join(dataset_folder, "dataset.hdf5")  # Path to HDF5 file

# Ensure dataset folder exists
os.makedirs(dataset_folder, exist_ok=True)

# Create HDF5 file
with h5py.File(hdf5_path, "w") as hdf5:
    # Create groups in HDF5
    raw_group = hdf5.create_group("raw")
    hdf5.create_group("pre_processed")
    hdf5.create_group("segmented")

    # Loop through CSV files in raw_data folder and store them in the 'raw' group
    for file in os.listdir(raw_data_folder):
        if file.endswith(".csv"):
            file_path = os.path.join(raw_data_folder, file)
            data = pd.read_csv(file_path)

            # Determine activity type from filename
            if "walking" in file.lower():
                activity = "walking"
            elif "jumping" in file.lower():
                activity = "jumping"
            else:
                activity = "unknown"

            # Create dataset path inside 'raw' group
            dataset_name = f"raw/{activity}/{file.replace('.csv', '')}"  # Store under 'walking' or 'running'

            # Ensure group exists for activity type
            if activity not in raw_group:
                raw_group.create_group(activity)

            # Store CSV data in HDF5 file
            raw_group[activity].create_dataset(file.replace(".csv", ""), data=data.to_numpy())

            print(f"Stored {file} in {dataset_name}")

print(f"Conversion complete. HDF5 file created at: {hdf5_path}")


dataset = pd.read_csv("C:\\Users\\charl\\.vscode\\290\\ELEC-292-Group-72\\Project\Project\\raw_data\\CharlotteWalking.csv")

acceleration_columns = ['Linear Acceleration x (m/s^2)', 'Linear Acceleration y (m/s^2)', 'Linear Acceleration z (m/s^2)', 'Absolute acceleration (m/s^2)']

dataset = dataset.astype(float)
#skip first row (labels)
time = dataset.iloc[1:, 0]
acceleration = dataset.iloc[1:, 1:]
#choose every 500th element
av_time = time.iloc[::500]
av_acceleration = acceleration.iloc[::500, :]

num_plots = len(acceleration_columns)
fig, ax = plt.subplots(ncols = 1, nrows = num_plots, figsize = ((5*num_plots),5), sharex = True)
for i, col in enumerate(av_acceleration.columns):
    ax[i].plot(av_time, av_acceleration[col], label = acceleration_columns[i], color = 'b', marker='o', linestyle='-')
    ax[i].set_xlabel("Time (s)")
    ax[i].set_ylabel(acceleration_columns[i])
    ax[i].set_title(acceleration_columns[i]+" vs Time")
    ax[i].grid(True)
    ax[i].legend()
fig.tight_layout()
plt.show()

# window_size = 5
moving_avg_dataset = pd.DataFrame()
for column in acceleration_columns:
#  #create a new column for the moving average of each acceleration type
      moving_avg_dataset[f'{column}_Moving_Avg'] = dataset[column].rolling(window=window_size).mean()
#  #show values for both raw and new moving average data
# # gui = show(moving_avg_dataset)

