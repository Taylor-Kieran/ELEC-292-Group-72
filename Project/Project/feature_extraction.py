import h5py
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.stats import skew, kurtosis
from scipy.fft import fft

def mad(x):
    return np.mean(np.abs(x - np.mean(x)))

# Zero Crossing Rate
def zero_crossings(x):
    return np.sum(np.diff(np.sign(x)) != 0)

# Autocorrelation (lag=1)
def autocorrelation(x):
    return np.corrcoef(x[:-1], x[1:])[0, 1]

# Peak-to-Peak Amplitude
def peak_to_peak(x):
    return np.max(x) - np.min(x)

# FFT (Frequency Domain Features)
def fft_features(x):
    fft_vals = fft(x)
    magnitude = np.abs(fft_vals)
    return np.mean(magnitude), np.std(magnitude), np.max(magnitude), np.min(magnitude)

# Function to extract 10 features from a 5-second segment of acceleration data
def extract_features_from_segment(data):
    """Extract features from a 5-second segment of acceleration data"""
    # Extract x, y, z, and absolute acceleration
    x, y, z, abs_accel = data[:, 0], data[:, 1], data[:, 2], data[:, 3]

    # Function to calculate RMS
    def rms(values):
        return np.sqrt(np.mean(values**2))

    # Function to calculate energy
    def energy(values):
        return np.sum(values**2)

    # Extract features (mean, std, max, min, range, median, rms, skewness, kurtosis, energy)
    return {
        # X-axis features
        'x_mean': np.mean(x),
                'x_std': np.std(x),
                'x_max': np.max(x),
                'x_min': np.min(x),
                'x_range': np.ptp(x),
                'x_median': np.median(x),
                'x_rms': rms(x),
                'x_skew': skew(x),
                'x_kurtosis': kurtosis(x),
                'x_energy': energy(x),
                'x_mad': mad(x),  # Mean Absolute Deviation
                'x_zcr': zero_crossings(x),  # Zero Crossing Rate
                'x_autocorr': autocorrelation(x),  # Autocorrelation
                'x_ptp': peak_to_peak(x),  # Peak to Peak Amplitude
                'x_fft_mean': fft_features(x)[0],  # FFT Mean
                'x_fft_std': fft_features(x)[1],  # FFT Standard Deviation
                'x_fft_max': fft_features(x)[2],  # FFT Max
                'x_fft_min': fft_features(x)[3],  # FFT Min

                # Y-axis features
                'y_mean': np.mean(y),
                'y_std': np.std(y),
                'y_max': np.max(y),
                'y_min': np.min(y),
                'y_range': np.ptp(y),
                'y_median': np.median(y),
                'y_rms': rms(y),
                'y_skew': skew(y),
                'y_kurtosis': kurtosis(y),
                'y_energy': energy(y),
                'y_mad': mad(y),
                'y_zcr': zero_crossings(y),
                'y_autocorr': autocorrelation(y),
                'y_ptp': peak_to_peak(y),
                'y_fft_mean': fft_features(y)[0],
                'y_fft_std': fft_features(y)[1],
                'y_fft_max': fft_features(y)[2],
                'y_fft_min': fft_features(y)[3],

                # Z-axis features
                'z_mean': np.mean(z),
                'z_std': np.std(z),
                'z_max': np.max(z),
                'z_min': np.min(z),
                'z_range': np.ptp(z),
                'z_median': np.median(z),
                'z_rms': rms(z),
                'z_skew': skew(z),
                'z_kurtosis': kurtosis(z),
                'z_energy': energy(z),
                'z_mad': mad(z),
                'z_zcr': zero_crossings(z),
                'z_autocorr': autocorrelation(z),
                'z_ptp': peak_to_peak(z),
                'z_fft_mean': fft_features(z)[0],
                'z_fft_std': fft_features(z)[1],
                'z_fft_max': fft_features(z)[2],
                'z_fft_min': fft_features(z)[3],

                # Absolute acceleration features
                'abs_mean': np.mean(abs_accel),
                'abs_std': np.std(abs_accel),
                'abs_max': np.max(abs_accel),
                'abs_min': np.min(abs_accel),
                'abs_range': np.ptp(abs_accel),
                'abs_median': np.median(abs_accel),
                'abs_rms': rms(abs_accel),
                'abs_skew': skew(abs_accel),
                'abs_kurtosis': kurtosis(abs_accel),
                'abs_energy': energy(abs_accel),
                'abs_mad': mad(abs_accel),
                'abs_zcr': zero_crossings(abs_accel),
                'abs_autocorr': autocorrelation(abs_accel),
                'abs_ptp': peak_to_peak(abs_accel),
                'abs_fft_mean': fft_features(abs_accel)[0],
                'abs_fft_std': fft_features(abs_accel)[1],
                'abs_fft_max': fft_features(abs_accel)[2],
                'abs_fft_min': fft_features(abs_accel)[3],
    }

# Function to extract features from pre-processed data
def feature_extraction(file_path):
    """Extract features from pre-processed acceleration data"""
    with h5py.File(file_path, "r") as f:
        preprocessed_group = f["pre-processed"]
        extracted_features = []

        count = 0
        for name in preprocessed_group:
            data = np.array(preprocessed_group[name])  # Assuming data is stored as a numpy array
            
            # Segment the data into 5-second windows (500 rows each)
            num_segments = len(data) // 500  # Number of full 5-second segments
            for i in range(num_segments):
                segment = data[i*500:(i+1)*500, :]
                
                # Extract features for the segment
                new_row = extract_features_from_segment(segment)
                
                # Determine label based on filename
                if "Walking" in name or "Jumping" in name:
                    label = 0 if "Walking" in name else 1
                    new_row['label'] = label  # Add the label to the features

                
                # Append the extracted features for this segment to the list
                extracted_features.append(new_row)
        # Convert the list of dictionaries to a DataFrame
        return pd.DataFrame(extracted_features)


# Normalize the feature data using Z-score normalization
from sklearn.preprocessing import StandardScaler
import pandas as pd

def normalize_features(extracted_df):
    scaler = StandardScaler()
    features = extracted_df.drop(columns=["label"], errors="ignore").values  # Drop 'label' if it exists
    normalized_features = scaler.fit_transform(features)
    
    normalized_df = pd.DataFrame(normalized_features, columns=[col for col in extracted_df.columns if col != "label"])
    
    # Reattach label only if it was originally present
    if "label" in extracted_df.columns:
        normalized_df["label"] = extracted_df["label"].values

    return normalized_df








