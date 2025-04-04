import h5py
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler


# Autocorrelation 
def autocorrelation(x):
    return np.corrcoef(x[:-1], x[1:])[0, 1]

# Function to extract 10 different features from a 5 second segment
def extract_features_from_segment(data):
    
    # Extracting features
    x, y, z, abs_accel = data[:, 0], data[:, 1], data[:, 2], data[:, 3]

    # Calculating RMS
    def rms(values):
        return np.sqrt(np.mean(values**2))

    # returning features
    return {
        
                'x_mean': np.mean(x),
                'x_std': np.std(x),
                'x_max': np.max(x),
                'x_min': np.min(x),
                'x_range': np.ptp(x),
                'x_median': np.median(x),
                'x_rms': rms(x),
                'x_skew': skew(x),
                'x_kurtosis': kurtosis(x),
                'x_autocorr': autocorrelation(x),  

                'y_mean': np.mean(y),
                'y_std': np.std(y),
                'y_max': np.max(y),
                'y_min': np.min(y),
                'y_range': np.ptp(y),
                'y_median': np.median(y),
                'y_rms': rms(y),
                'y_skew': skew(y),
                'y_kurtosis': kurtosis(y),
                'y_autocorr': autocorrelation(y),
       
                'z_mean': np.mean(z),
                'z_std': np.std(z),
                'z_max': np.max(z),
                'z_min': np.min(z),
                'z_range': np.ptp(z),
                'z_median': np.median(z),
                'z_rms': rms(z),
                'z_skew': skew(z),
                'z_kurtosis': kurtosis(z),
                'z_autocorr': autocorrelation(z),

                'abs_mean': np.mean(abs_accel),
                'abs_std': np.std(abs_accel),
                'abs_max': np.max(abs_accel),
                'abs_min': np.min(abs_accel),
                'abs_range': np.ptp(abs_accel),
                'abs_median': np.median(abs_accel),
                'abs_rms': rms(abs_accel),
                'abs_skew': skew(abs_accel),
                'abs_kurtosis': kurtosis(abs_accel),
                'abs_autocorr': autocorrelation(abs_accel),
                
    }

# Function to extract features from pre-processed data
def feature_extraction(file_path):
    with h5py.File(file_path, "r") as f:
        preprocessed_group = f["pre-processed"]
        extracted_features = []
        
        for name in preprocessed_group:
            data = np.array(preprocessed_group[name])  
            
            # Segmenting the data into 5 second segments
            num_segments = len(data) // 500  
            for i in range(num_segments):
                segment = data[i*500:(i+1)*500, :]
                # Extracting features for the segment
                new_row = extract_features_from_segment(segment)
                
                # Determining label based on filename
                if "Walking" in name or "Jumping" in name:
                    label = 0 if "Walking" in name else 1
                    new_row['label'] = label  

                
                # Appending the dataframes
                extracted_features.append(new_row)

        return pd.DataFrame(extracted_features)


# Normalizing the feature data using Z-score normalization
def normalize_features(extracted_df):
    scaler = StandardScaler()
    features = extracted_df.drop(columns=["label"], errors="ignore").values 
    normalized_features = scaler.fit_transform(features)
    
    normalized_df = pd.DataFrame(normalized_features, columns=[col for col in extracted_df.columns if col != "label"])
    
    # Reattaching the labels
    if "label" in extracted_df.columns:
        normalized_df["label"] = extracted_df["label"].values

    return normalized_df








