import h5py
import pandas as pd
import matplotlib.pyplot as plt

file_path = "Project/Project/dataset/dataset.hdf5"

with h5py.File(file_path, "r") as f:
    if 'raw' in f:

        datasets = list(f['raw'].keys())
        print("Datasets in 'raw' group:", datasets)
    else:
        raise FileNotFoundError("'raw' group not found in the dataset.")

# Checking for datasets
if len(datasets) == 0:
    raise ValueError("No datasets found in the 'raw' group.")

# storing walking and jumping data
walking_data = {"x": [], "y": [], "z": [], "abs": []}
jumping_data = {"x": [], "y": [], "z": [], "abs": []}

# Iterating through the datasets
for dataset_name in datasets:
    try:
        with h5py.File(file_path, "r") as f:
            data = f['raw'][dataset_name][:]
        
        # Converting to dataframe
        columns = ["Time (s)", "Linear Acceleration x (m/s^2)", "Linear Acceleration y (m/s^2)", 
                   "Linear Acceleration z (m/s^2)", "Absolute acceleration (m/s^2)"]
        df = pd.DataFrame(data, columns=columns)

        # 
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

# Combining walking and jumping data
walking_data = {key: pd.concat(val, ignore_index=True) for key, val in walking_data.items()}
jumping_data = {key: pd.concat(val, ignore_index=True) for key, val in jumping_data.items()}

# creating a continous plot
fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(16, 8))

# walking data
ax[0, 0].hist(walking_data["x"], bins=30, density=True, histtype='step', color='b')
ax[0, 0].set_title("1. Walking - X Acceleration")
ax[0, 0].set_xlabel("Acceleration (m/s^2)")
ax[0, 0].set_ylabel("Density")

ax[0, 1].hist(walking_data["y"], bins=30, density=True, histtype='step', color='g')
ax[0, 1].set_title("2. Walking - Y Acceleration")
ax[0, 1].set_xlabel("Acceleration (m/s^2)")
ax[0, 1].set_ylabel("Density")

ax[0, 2].hist(walking_data["z"], bins=30, density=True, histtype='step', color='r')
ax[0, 2].set_title("3. Walking - Z Acceleration")
ax[0, 2].set_xlabel("Acceleration (m/s^2)")
ax[0, 2].set_ylabel("Density")

ax[0, 3].hist(walking_data["abs"], bins=30, density=True, histtype='step', color='purple')
ax[0, 3].set_title("4. Walking - Absolute Acceleration")
ax[0, 3].set_xlabel("Acceleration (m/s^2)")
ax[0, 3].set_ylabel("Density")

# Jumping data
ax[1, 0].hist(jumping_data["x"], bins=30, density=True, histtype='step', color='b')
ax[1, 0].set_title("5. Jumping - X Acceleration")
ax[1, 0].set_xlabel("6. Acceleration (m/s^2)")
ax[1, 0].set_ylabel("Density")

ax[1, 1].hist(jumping_data["y"], bins=30, density=True, histtype='step', color='g')
ax[1, 1].set_title("6. Jumping - Y Acceleration")
ax[1, 1].set_xlabel("Acceleration (m/s^2)")
ax[1, 1].set_ylabel("Density")

ax[1, 2].hist(jumping_data["z"], bins=30, density=True, histtype='step', color='r')
ax[1, 2].set_title("7. Jumping - Z Acceleration")
ax[1, 2].set_xlabel("Acceleration (m/s^2)")
ax[1, 2].set_ylabel("Density")

ax[1, 3].hist(jumping_data["abs"], bins=30, density=True, histtype='step', color='purple')
ax[1, 3].set_title("8. Jumping - Absolute Acceleration")
ax[1, 3].set_xlabel("Acceleration (m/s^2)")
ax[1, 3].set_ylabel("Density")


plt.tight_layout()
plt.show()
