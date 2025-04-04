import h5py
import numpy as np
import pandas as pd

# Moving Average Filter
def moving_average(data, window_size=50):
    return np.convolve(data, np.ones(window_size)/window_size, mode='same')

# Function to process a dataset
def process_dataset(data):
    # Converting to DataFrame for easier processing
    df = pd.DataFrame(data)
    
    # filling in missing values using forward and backward fill
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    
    # Appling the moving average filter
    for col in df.columns[1:]:
        df[col] = moving_average(df[col])
    return df

# Function to process HDF5 file
def process_hdf5(file_path):
    with h5py.File(file_path, "r+") as f:
        raw_group = f["raw"]
        
        # Creating pre-processed group if not exists
        if "pre-processed" not in f:
            f.create_group("pre-processed")
        
        processed_data = {}
        
        for name in raw_group:
            data = np.array(raw_group[name])  
            processed_data[name] = process_dataset(data)  
        
    return processed_data


