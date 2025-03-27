import h5py
import pandas as pd
import numpy as np
import os

# Define file paths
hdf5_file_path = "Project/Project/dataset/dataset.hdf5"
output_walking_csv = "walking.csv"
output_jumping_csv = "jumping.csv"

# Lists to store combined data
walking_data = []
jumping_data = []

# Open the HDF5 file
with h5py.File(hdf5_file_path, "r") as f:
    if "pre-processed" not in f:
        raise FileNotFoundError("'pre-processed' group not found in HDF5 file.")
    
    for file_name in f["pre-processed"]:
        csv_data = pd.DataFrame(f["pre-processed"][file_name][:])
        
        # Check if file name contains 'Jumping' or 'Walking'
        if "Jumping" in file_name:
            jumping_data.append(csv_data)
        elif "Walking" in file_name:
            walking_data.append(csv_data)

# Combine all data for walking and jumping
if walking_data:
    walking_df = pd.concat(walking_data, ignore_index=True)
    walking_df.to_csv(output_walking_csv, index=False)
    print(f"Walking data saved to {output_walking_csv}")

if jumping_data:
    jumping_df = pd.concat(jumping_data, ignore_index=True)
    jumping_df.to_csv(output_jumping_csv, index=False)
    print(f"Jumping data saved to {output_jumping_csv}")
