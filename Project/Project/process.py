import h5py
import numpy as np
import pandas as pd

# Moving Average Filter
def moving_average(data, window_size=50):
    return np.convolve(data, np.ones(window_size)/window_size, mode='same')

# Function to process a dataset
def process_dataset(data):
    # Convert to DataFrame for easier processing
    df = pd.DataFrame(data)
    
    # filling in missing values using forward and backward fill
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    
    # Apply moving average filter to all columns except the first (time column)
    for col in df.columns[1:]:  # Skip time column (assuming column 0 is time)
        df[col] = moving_average(df[col])
    return df

# Function to process HDF5 file
def process_hdf5(file_path):
    with h5py.File(file_path, "r+") as f:
        raw_group = f["raw"]
        
        # Create pre-processed group if not exists
        if "pre-processed" not in f:
            f.create_group("pre-processed")
        
        processed_data = {}
        
        for name in raw_group:
            data = np.array(raw_group[name])  # Load dataset
            processed_data[name] = process_dataset(data)  # Process data
        
    return processed_data


