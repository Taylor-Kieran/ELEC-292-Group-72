import pandas as pd
import numpy as np

# Define columns
columns = [
    # X-axis (10 features)
    'x_mean', 'x_std', 'x_max', 'x_min', 'x_range', 
    'x_median', 'x_rms', 'x_skew', 'x_kurtosis', 'x_zcr',
    
    # Y-axis (10 features)
    'y_mean', 'y_std', 'y_max', 'y_min', 'y_range',
    'y_median', 'y_rms', 'y_skew', 'y_kurtosis', 'y_zcr',
    
    # Z-axis (10 features)
    'z_mean', 'z_std', 'z_max', 'z_min', 'z_range',
    'z_median', 'z_rms', 'z_skew', 'z_kurtosis', 'z_zcr',
    
    # Absolute acceleration (10 features)
    'abs_mean', 'abs_std', 'abs_max', 'abs_min', 'abs_range',
    'abs_median', 'abs_rms', 'abs_skew', 'abs_kurtosis', 'abs_zcr',
    
    # Label
    'label'
]

# Create empty DataFrame
extracted_features = pd.DataFrame(columns=columns)

# Generate and add 3 random rows of data
for _ in range(3):
    # Random values for features (between 0 and 2 for demonstration)
    random_features = np.round(np.random.uniform(0, 2, 40), 3)
    
    # Random label (0 or 1)
    random_label = np.random.randint(0, 2)
    
    # Combine into a row
    row_data = np.append(random_features, random_label)
    
    # Add to DataFrame
    extracted_features.loc[len(extracted_features)] = row_data

# Print the DataFrame
print("DataFrame with random values:")
print(extracted_features)

# Print transposed for better readability of all columns
print("\nTransposed view (showing all columns):")
print(extracted_features.T)