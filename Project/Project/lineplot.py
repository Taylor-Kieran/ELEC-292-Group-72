import h5py
import pandas as pd
import matplotlib.pyplot as plt

# Define the path to your HDF5 file
file_path = "Project/Project/dataset/dataset.hdf5"

# Open the HDF5 file and check if the 'raw' group exists
with h5py.File(file_path, "r") as f:
    if 'raw' in f:
        datasets = [ds for ds in f['raw'].keys() if "Charlotte" not in ds]  # Exclude "Charlotte" datasets
        print("Filtered datasets in 'raw' group:", datasets)
    else:
        raise FileNotFoundError("'raw' group not found in the dataset.")

if len(datasets) == 0:
    raise ValueError("No valid datasets found in the 'raw' group after filtering.")

# Store walking and jumping data
walking_data = {"x": [], "y": [], "z": [], "abs": []}
jumping_data = {"x": [], "y": [], "z": [], "abs": []}

# Iterate through all datasets in the 'raw' group
for dataset_name in datasets:
    try:
        with h5py.File(file_path, "r") as f:
            data = f['raw'][dataset_name][:]

        # Convert to DataFrame
        columns = ["Time (s)", "Linear Acceleration x (m/s^2)", "Linear Acceleration y (m/s^2)", 
                   "Linear Acceleration z (m/s^2)", "Absolute acceleration (m/s^2)"]
        df = pd.DataFrame(data, columns=columns)

        # Store the time series data
        if 'Walking' in dataset_name:
            walking_data["x"].append(df["Linear Acceleration x (m/s^2)"])
            walking_data["y"].append(df["Linear Acceleration y (m/s^2)"])
            walking_data["z"].append(df["Linear Acceleration z (m/s^2)"])
            walking_data["abs"].append(df["Absolute acceleration (m/s^2)"])
        elif 'Jumping' in dataset_name:
            jumping_data["x"].append(df["Linear Acceleration x (m/s^2)"])
            jumping_data["y"].append(df["Linear Acceleration y (m/s^2)"])
            jumping_data["z"].append(df["Linear Acceleration z (m/s^2)"])
            jumping_data["abs"].append(df["Absolute acceleration (m/s^2)"])

    except Exception as e:
        print(f"Failed to process dataset {dataset_name}: {e}")

# Plot settings
fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(16, 8))

# Function to plot all lines for a given category
def plot_lines(ax, data, title, color):
    for series in data:
        ax.plot(series, alpha=0.5, color=color, linewidth=0.8)  # Semi-transparent overlapping lines
    ax.set_title(title)
    ax.set_xlabel("Time (samples)")
    ax.set_ylabel("Acceleration (m/s^2)")

# Walking Line Plots
plot_lines(ax[0, 0], walking_data["x"], "1. Walking - X Acceleration", "b")
plot_lines(ax[0, 1], walking_data["y"], "2. Walking - Y Acceleration", "g")
plot_lines(ax[0, 2], walking_data["z"], "3. Walking - Z Acceleration", "r")
plot_lines(ax[0, 3], walking_data["abs"], "4. Walking - Absolute Acceleration", "purple")

# Jumping Line Plots
plot_lines(ax[1, 0], jumping_data["x"], "5. Jumping - X Acceleration", "b")
plot_lines(ax[1, 1], jumping_data["y"], "6. Jumping - Y Acceleration", "g")
plot_lines(ax[1, 2], jumping_data["z"], "7. Jumping - Z Acceleration", "r")
plot_lines(ax[1, 3], jumping_data["abs"], "8. Jumping - Absolute Acceleration", "purple")

# Adjust layout
plt.tight_layout()
plt.show()
