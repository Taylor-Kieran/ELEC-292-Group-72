import h5py
import pandas as pd
import matplotlib.pyplot as plt

# Define file path
file_path = "Project/Project/dataset/dataset.hdf5"

# Ask user for dataset group and dataset name
group_name = input("Enter group name ('raw' or 'pre-processed'): ").strip()
dataset_name = input("Enter dataset name to visualize: ").strip()

# Ask user whether to display line graphs or histograms
plot_type = input("Would you like to see line graphs or histograms? (Enter 'line' or 'histogram'): ").strip().lower()

# Open the HDF5 file and check if dataset exists
with h5py.File(file_path, "r") as f:
    if group_name in f and dataset_name in f[group_name]:
        data = f[group_name][dataset_name][:]  # Load the dataset
    else:
        raise FileNotFoundError(f"Dataset '{dataset_name}' not found in '{group_name}' group.")

# Convert to DataFrame
columns = ["Time (s)", "Linear Acceleration x (m/s^2)", "Linear Acceleration y (m/s^2)", 
           "Linear Acceleration z (m/s^2)", "Absolute acceleration (m/s^2)"]
df = pd.DataFrame(data, columns=columns)

# Extract time and acceleration values
time = df["Time (s)"]
acceleration_columns = columns[1:]

# Create the plots based on user choice
fig, ax = plt.subplots(ncols=1, nrows=len(acceleration_columns), figsize=(10, 8), sharex=False)

if plot_type == 'line':
    for i, col in enumerate(acceleration_columns):
        ax[i].plot(time, df[col], label=col, linestyle='-')
        ax[i].set_ylabel(col)
        ax[i].set_title(f"{col} vs Time")
        ax[i].grid(True)
        ax[i].legend()
elif plot_type == 'histogram':
    # Use the DataFrame .plot method to create density plots (continuous histograms)
    df[acceleration_columns].plot(ax=ax.flatten(), kind='density', subplots=True, sharex=False)
    for i, col in enumerate(acceleration_columns):
        ax[i].set_ylabel("Density")
        ax[i].set_title(f"Continuous Histogram (Density) of {col}")
    
fig.tight_layout()
plt.show()
