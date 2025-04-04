import h5py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Define the path to your HDF5 file
file_path = "Project/Project/dataset/dataset.hdf5"

# Open the HDF5 file and check if the 'raw' group exists
with h5py.File(file_path, "r") as f:
    if 'raw' in f:
        # List all datasets in the 'raw' group to confirm what exists
        datasets = list(f['raw'].keys())
        print("Datasets in 'raw' group:", datasets)
    else:
        raise FileNotFoundError("'raw' group not found in the dataset.")

# Check if there are any datasets
if len(datasets) == 0:
    raise ValueError("No datasets found in the 'raw' group.")

# Lists to store walking and jumping data
walking_data = {"x": [], "y": [], "z": [], "abs": []}
jumping_data = {"x": [], "y": [], "z": [], "abs": []}

# Iterate through all datasets in the 'raw' group
for dataset_name in datasets:
    try:
        # Load the dataset from the 'raw' group
        with h5py.File(file_path, "r") as f:
            data = f['raw'][dataset_name][:]

        # Convert to DataFrame
        columns = ["Time (s)", "Linear Acceleration x (m/s^2)", "Linear Acceleration y (m/s^2)", 
                   "Linear Acceleration z (m/s^2)", "Absolute acceleration (m/s^2)"]
        df = pd.DataFrame(data, columns=columns)

        # Filter based on the filename (dataset name) containing "Walking" or "Jumping"
        if 'Walking' in dataset_name:
            walking_data["x"].append(df['Linear Acceleration x (m/s^2)'])
            walking_data["y"].append(df['Linear Acceleration y (m/s^2)'])
            walking_data["z"].append(df['Linear Acceleration z (m/s^2)'])
            walking_data["abs"].append(df['Absolute acceleration (m/s^2)'])
        elif 'Jumping' in dataset_name:
            jumping_data["x"].append(df['Linear Acceleration x (m/s^2)'])
            jumping_data["y"].append(df['Linear Acceleration y (m/s^2)'])
            jumping_data["z"].append(df['Linear Acceleration z (m/s^2)'])
            jumping_data["abs"].append(df['Absolute acceleration (m/s^2)'])

    except Exception as e:
        print(f"Failed to process dataset {dataset_name}: {e}")

# Combine all walking and jumping data
walking_data = {key: pd.concat(val, ignore_index=True) for key, val in walking_data.items()}
jumping_data = {key: pd.concat(val, ignore_index=True) for key, val in jumping_data.items()}

# Create DataFrames for Walking and Jumping data
walking_df = pd.DataFrame(walking_data)
jumping_df = pd.DataFrame(jumping_data)

# Calculate the correlation matrices for both Walking and Jumping data
walking_corr = walking_df.corr()
jumping_corr = jumping_df.corr()

# Plot the correlation matrices using seaborn heatmap
fig, ax = plt.subplots(1, 2, figsize=(16, 6))

# Walking Correlation Matrix
sns.heatmap(walking_corr, annot=True, cmap='coolwarm', ax=ax[0], cbar=True, vmin=-1, vmax=1)
ax[0].set_title("Walking Data - Correlation Matrix")

# Jumping Correlation Matrix
sns.heatmap(jumping_corr, annot=True, cmap='coolwarm', ax=ax[1], cbar=True, vmin=-1, vmax=1)
ax[1].set_title("Jumping Data - Correlation Matrix")

# Adjust layout
plt.tight_layout()
plt.show()
