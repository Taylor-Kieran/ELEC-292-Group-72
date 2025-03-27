import h5py
import pandas as pd
import numpy as np

# Configuration
SAMPLING_RATE = 100  # Hz
WINDOW_SECONDS = 5
WINDOW_SIZE = SAMPLING_RATE * WINDOW_SECONDS
HDF5_PATH = "Project/Project/dataset/dataset.hdf5"
OUTPUT_CSV_NAME = "extracted.csv"

# Column definitions (same as before)
columns = [
    # X-axis features (10)
    'x_mean', 'x_std', 'x_max', 'x_min', 'x_range', 
    'x_median', 'x_rms', 'x_skew', 'x_kurtosis', 'x_zcr',
    # Y-axis features (10)
    'y_mean', 'y_std', 'y_max', 'y_min', 'y_range',
    'y_median', 'y_rms', 'y_skew', 'y_kurtosis', 'y_zcr',
    # Z-axis features (10)
    'z_mean', 'z_std', 'z_max', 'z_min', 'z_range',
    'z_median', 'z_rms', 'z_skew', 'z_kurtosis', 'z_zcr',
    # Absolute acceleration features (10)
    'abs_mean', 'abs_std', 'abs_max', 'abs_min', 'abs_range',
    'abs_median', 'abs_rms', 'abs_skew', 'abs_kurtosis', 'abs_zcr',
    # Label
    'label'
]

# Create empty DataFrame
extracted_features = pd.DataFrame(columns=columns)

def process_segment(segment, file_name, dataframe):
    """Process a 5-second segment and append features to dataframe"""
    x, y, z, abs_accel = segment[:, 0], segment[:, 1], segment[:, 2], segment[:, 3]
    
    new_row = {
        # X-axis features
        'x_mean': np.mean(x),
        'x_std': np.std(x),
        'x_max': np.max(x),
        'x_min': np.min(x),
        'x_range': np.ptp(x),
        'x_median': np.median(x),
        'x_rms': np.sqrt(np.mean(x**2)),
        'x_skew': pd.Series(x).skew(),
        'x_kurtosis': pd.Series(x).kurtosis(),
        'x_zcr': np.sum(np.diff(np.sign(x)) != 0)/len(x),
        
        # Y-axis features (same pattern as x)
        'y_mean': np.mean(y),
        'y_std': np.std(y),
        'y_max': np.max(y),
        'y_min': np.min(y),
        'y_range': np.ptp(y),
        'y_median': np.median(y),
        'y_rms': np.sqrt(np.mean(y**2)),
        'y_skew': pd.Series(y).skew(),
        'y_kurtosis': pd.Series(y).kurtosis(),
        'y_zcr': np.sum(np.diff(np.sign(y)) != 0)/len(y),
        
        # Z-axis features (same pattern)
        'z_mean': np.mean(z),
        'z_std': np.std(z),
        'z_max': np.max(z),
        'z_min': np.min(z),
        'z_range': np.ptp(z),
        'z_median': np.median(z),
        'z_rms': np.sqrt(np.mean(z**2)),
        'z_skew': pd.Series(z).skew(),
        'z_kurtosis': pd.Series(z).kurtosis(),
        'z_zcr': np.sum(np.diff(np.sign(z)) != 0)/len(z),
        
        # Absolute acceleration features
        'abs_mean': np.mean(abs_accel),
        'abs_std': np.std(abs_accel),
        'abs_max': np.max(abs_accel),
        'abs_min': np.min(abs_accel),
        'abs_range': np.ptp(abs_accel),
        'abs_median': np.median(abs_accel),
        'abs_rms': np.sqrt(np.mean(abs_accel**2)),
        'abs_skew': pd.Series(abs_accel).skew(),
        'abs_kurtosis': pd.Series(abs_accel).kurtosis(),
        'abs_zcr': np.sum(np.diff(np.sign(abs_accel)) != 0)/len(abs_accel),
        
        # Label
        'label': 1 if "Jumping" in file_name else 0
    }
    
    dataframe.loc[len(dataframe)] = new_row

# Main processing
with h5py.File(HDF5_PATH, "r") as f:  # Read-only mode
    pre_processed_group = f["pre-processed"]
    
    for file_name in pre_processed_group:
        # Get the dataset (CSV data stored in HDF5)
        dataset = pre_processed_group[file_name]
        
        # Convert to numpy array (assuming data is stored as array)
        data = np.array(dataset)
        
        # Process in 5-second windows
        num_segments = len(data) // WINDOW_SIZE
        
        for i in range(num_segments):
            start_idx = i * WINDOW_SIZE
            end_idx = start_idx + WINDOW_SIZE
            segment = data[start_idx:end_idx, 1:5]  # Skip time column (col 0)
            
            process_segment(segment, file_name, extracted_features)

# Display the resulting DataFrame




with h5py.File(HDF5_PATH, "a") as f:  # 'a' mode to append without overwriting
    # Create segmented group if it doesn't exist
    if "segmented" not in f:
        segmented_group = f.create_group("segmented")
    else:
        segmented_group = f["segmented"]
    
    # Remove existing dataset if present (to avoid conflicts)
    if OUTPUT_CSV_NAME in segmented_group:
        del segmented_group[OUTPUT_CSV_NAME]
    
    # Convert DataFrame to CSV string then to bytes
    csv_data = extracted_features.to_csv(index=False)
    csv_bytes = csv_data.encode('utf-8')
    
    # Store in HDF5 as a fixed-length string dataset
    segmented_group.create_dataset(
        OUTPUT_CSV_NAME,
        data=csv_bytes,
        dtype=h5py.string_dtype('utf-8')
    )

print(f"\nCSV dataset successfully saved to: /segmented/{OUTPUT_CSV_NAME}")
print("Final DataFrame summary:")
print(extracted_features.info())
