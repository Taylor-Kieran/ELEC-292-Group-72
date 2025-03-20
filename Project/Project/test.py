import os
import pandas as pd
import h5py

# Get absolute paths relative to the script's location
project_folder = os.path.dirname(os.path.abspath(__file__))  # Path to project folder
raw_data_folder = os.path.join(project_folder, "raw_data")  # Folder containing raw CSV files
dataset_folder = os.path.join(project_folder, "dataset")  # Folder where dataset.hdf5 is stored
hdf5_path = os.path.join(dataset_folder, "dataset.hdf5")  # Path to HDF5 file

# Only create dataset folder if it does not exist
if not os.path.exists(dataset_folder):
    os.makedirs(dataset_folder)  # Create dataset folder only if missing

# Open HDF5 file in append mode
with h5py.File(hdf5_path, "a") as hdf5:
    # Ensure 'raw' group exists
    raw_group = hdf5.require_group("raw")  # Creates if missing, else uses existing

    # Loop through CSV files in raw_data folder and store them in 'raw' group
    for file in os.listdir(raw_data_folder):
        if file.endswith(".csv"):
            file_path = os.path.join(raw_data_folder, file)
            data = pd.read_csv(file_path)

            dataset_name = file.replace(".csv", "")  # Name of dataset in HDF5

            # Check if dataset already exists in 'raw'
            if dataset_name in raw_group:
                print(f"Skipping {file}, already exists in /raw/")
                continue  # Skip if already present

            # Store CSV data in HDF5 file inside 'raw' group
            raw_group.create_dataset(dataset_name, data=data.to_numpy())

            print(f"Stored {file} in /raw/{dataset_name}")

print(f"Update complete. HDF5 file updated at: {hdf5_path}")
