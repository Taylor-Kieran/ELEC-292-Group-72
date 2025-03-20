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

# Debug: Check if raw_data exists
if not os.path.exists(raw_data_folder):
    raise FileNotFoundError(f"Folder '{raw_data_folder}' not found in {os.getcwd()}")

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

            # Ensure group exists for activity type
            if activity not in hdf5["raw"]:
                hdf5["raw"].create_group(activity)

            # Store CSV data in HDF5 file
            hdf5["raw"][activity].create_dataset(file.replace(".csv", ""), data=data.to_numpy())

            print(f"Stored {file} in /raw/{activity}/{file.replace('.csv', '')}")

print(f"Conversion complete. HDF5 file created at: {hdf5_path}")
