import h5py
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler  # or MinMaxScaler

# Moving Average Filter
def moving_average(data, window_size=50):
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
        # Fill missing values using forward fill, then backward fill
        df.ffill(inplace=True)  # Replaces df.fillna(method='ffill', inplace=True)
        df.bfill(inplace=True)  # Replaces df.fillna(method='bfill', inplace=True)
        
        # Apply moving average filter to all columns except the first (time column)
        for col in df.columns[1:]:  # Skip time column (assuming column 0 is time)
            df[col] = moving_average(df[col])
        
        # --- NORMALIZATION STEP ---
        # Separate time column (if needed) and normalize the rest
        time_column = df.iloc[:, 0]  # Save time column (assuming it's column 0)
        data_to_normalize = df.iloc[:, 1:]  # Select only X/Y/Z axes (or other sensor data)
        
        # Initialize scaler (StandardScaler or MinMaxScaler)
        scaler = StandardScaler()  # Or MinMaxScaler(feature_range=(-1, 1))
        normalized_data = scaler.fit_transform(data_to_normalize)
        
        # Recombine time column with normalized data
        df_normalized = pd.DataFrame(normalized_data, columns=data_to_normalize.columns)
        df_normalized.insert(0, df.columns[0], time_column)  # Reinsert time column"""
        
        # Save to pre-processed group
        if name in preprocessed_group:
            del preprocessed_group[name]  # Remove existing dataset if present
        preprocessed_group.create_dataset(name, data=df_normalized.to_numpy())
    
    print("Pre-processing complete. Normalized data saved in 'pre-processed' group.")