import h5py
import pandas as pd
import matplotlib.pyplot as plt

# Load the HDF5 file
hdf5_path = "dataset/dataset.hdf5"

with h5py.File(hdf5_path, "r") as hdf5:
    activity = "raw/walking/Fateen_Walking_FrontPocket"  # Modify this to change datasets

    if activity in hdf5:
        data = hdf5[activity][()]

        # Convert to DataFrame
        df = pd.DataFrame(data, columns=["timestamp", "x", "y", "z", "absolute_acceleration"])

        # Plot x-axis acceleration over time
        plt.figure(figsize=(10, 4))
        plt.plot(df["timestamp"], df["x"], label="X-axis Acceleration", color="blue")
        plt.xlabel("Timestamp")
        plt.ylabel("X Acceleration")
        plt.title("X-axis Acceleration Over Time")
        plt.legend()
        plt.grid(True)
        plt.show()

    else:
        print(f"Dataset {activity} not found in HDF5 file.")

