import h5py
import numpy as np
import pandas as pd

# Moving Average Filter
def moving_average(data, window_size=5):
    return np.convolve(data, np.ones(window_size)/window_size, mode='same')

# Open the HDF5 file
file_path = "Project/Project/dataset/dataset.hdf5"
with h5py.File(file_path, "r+") as f:
    raw_group = f["raw"]
    
    # Create pre-processed group if not exists
    if "pre-processed" not in f:
        preprocessed_group = f.create_group("pre-processed")
    else:
        preprocessed_group = f["pre-processed"]
    
    for name in raw_group:
        data = np.array(raw_group[name])  # Load dataset
        
        # Convert to DataFrame for easier processing
        df = pd.DataFrame(data)
        
        # Fill missing values using forward fill, then backward fill
        df.fillna(method='ffill', inplace=True)
        df.fillna(method='bfill', inplace=True)
        
        # Apply moving average filter to smooth each column
        df_smoothed = df.apply(lambda col: moving_average(col), axis=0)
        
        # Save to pre-processed group
        if name in preprocessed_group:
            del preprocessed_group[name]  # Remove existing dataset if present
        preprocessed_group.create_dataset(name, data=df_smoothed.to_numpy())
    
    print("Pre-processing complete. Data saved in 'pre-processed' group.")
