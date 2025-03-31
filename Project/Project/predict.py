import numpy as np
import pandas as pd
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Configuration
SAMPLING_RATE = 100  # Hz
WINDOW_SECONDS = 5
WINDOW_SIZE = SAMPLING_RATE * WINDOW_SECONDS


def load_csv(file_path):
    """Load CSV file and return as DataFrame."""
    df = pd.read_csv(file_path)
    df.ffill(inplace=True)  # Forward fill missing values
    df.bfill(inplace=True)  # Backward fill missing values
    return df

def preprocess_data(df):
    """Applies smoothing and normalization to the sensor data."""
    def moving_average(data, window_size=50):
        return np.convolve(data, np.ones(window_size)/window_size, mode='same')

    # Apply moving average
    for col in df.columns[1:]:  # Skip the first column (time)
        df[col] = moving_average(df[col])

    # Normalize sensor data
    time_column = df.iloc[:, 0]  # First column is assumed to be time
    data_to_normalize = df.iloc[:, 1:]  # Exclude time column
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(data_to_normalize)

    # Recombine into DataFrame
    df_normalized = pd.DataFrame(normalized_data, columns=data_to_normalize.columns)
    df_normalized.insert(0, df.columns[0], time_column)
    return df_normalized

def extract_features(df):
    """Extracts statistical features from accelerometer data in 5-second windows."""
    columns = [
        'x_mean', 'x_std', 'x_max', 'x_min', 'x_range', 'x_median', 'x_rms', 'x_skew', 'x_kurtosis', 'x_zcr',
        'y_mean', 'y_std', 'y_max', 'y_min', 'y_range', 'y_median', 'y_rms', 'y_skew', 'y_kurtosis', 'y_zcr',
        'z_mean', 'z_std', 'z_max', 'z_min', 'z_range', 'z_median', 'z_rms', 'z_skew', 'z_kurtosis', 'z_zcr',
        'abs_mean', 'abs_std', 'abs_max', 'abs_min', 'abs_range', 'abs_median', 'abs_rms', 'abs_skew', 'abs_kurtosis', 'abs_zcr'
    ]

    extracted_features = pd.DataFrame(columns=columns)

    def compute_features(segment):
        """Computes statistical features for a given data segment."""
        x, y, z = segment[:, 0], segment[:, 1], segment[:, 2]
        abs_accel = np.sqrt(x**2 + y**2 + z**2)

        return {
            'x_mean': np.mean(x), 'x_std': np.std(x), 'x_max': np.max(x), 'x_min': np.min(x), 'x_range': np.ptp(x),
            'x_median': np.median(x), 'x_rms': np.sqrt(np.mean(x**2)), 'x_skew': pd.Series(x).skew(), 'x_kurtosis': pd.Series(x).kurtosis(),
            'x_zcr': np.sum(np.diff(np.sign(x)) != 0) / len(x),

            'y_mean': np.mean(y), 'y_std': np.std(y), 'y_max': np.max(y), 'y_min': np.min(y), 'y_range': np.ptp(y),
            'y_median': np.median(y), 'y_rms': np.sqrt(np.mean(y**2)), 'y_skew': pd.Series(y).skew(), 'y_kurtosis': pd.Series(y).kurtosis(),
            'y_zcr': np.sum(np.diff(np.sign(y)) != 0) / len(y),

            'z_mean': np.mean(z), 'z_std': np.std(z), 'z_max': np.max(z), 'z_min': np.min(z), 'z_range': np.ptp(z),
            'z_median': np.median(z), 'z_rms': np.sqrt(np.mean(z**2)), 'z_skew': pd.Series(z).skew(), 'z_kurtosis': pd.Series(z).kurtosis(),
            'z_zcr': np.sum(np.diff(np.sign(z)) != 0) / len(z),

            'abs_mean': np.mean(abs_accel), 'abs_std': np.std(abs_accel), 'abs_max': np.max(abs_accel), 'abs_min': np.min(abs_accel), 'abs_range': np.ptp(abs_accel),
            'abs_median': np.median(abs_accel), 'abs_rms': np.sqrt(np.mean(abs_accel**2)), 'abs_skew': pd.Series(abs_accel).skew(), 'abs_kurtosis': pd.Series(abs_accel).kurtosis(),
            'abs_zcr': np.sum(np.diff(np.sign(abs_accel)) != 0) / len(abs_accel),
        }

    # Process in 5-second windows
    data = df.iloc[:, 1:4].to_numpy()  # Exclude time column
    num_segments = len(data) // WINDOW_SIZE

    for i in range(num_segments):
        start_idx = i * WINDOW_SIZE
        end_idx = start_idx + WINDOW_SIZE
        segment = data[start_idx:end_idx, :]
        
        extracted_features.loc[len(extracted_features)] = compute_features(segment)

    return extracted_features


file_path = "input"

def predict(file_path):
    
    df = load_csv(file_path)

    # Data Processing
    df_processed = preprocess_data(df)

    # Feature Extraction
    df_features = extract_features(df_processed)

    # Prediction Model
    # Load the trained model
    model_path = "C://Users//Kieran Taylor//Documents//GitHub//ELEC-292-Group-72//trained_model.pkl"
    clf = joblib.load(model_path)
    print(f"Model loaded from {model_path}")

    # Load the new unlabeled dataset
    unlabeled_test_set = df_features

    # Apply the same feature scaling
    X_unlabeled = clf.named_steps["standardscaler"].transform(unlabeled_test_set.values)

    # Make predictions
    predicted_labels = clf.predict(X_unlabeled)
    predicted_probs = clf.predict_proba(X_unlabeled)[:, 1]  # Probability of jumping

    # Save results to CSV
    output_df = pd.DataFrame({
        "Sample_ID": np.arange(len(predicted_labels)),  # Optional index
        "Predicted_Label": predicted_labels,  # 0 = walking, 1 = jumping
        "Jumping_Probability": predicted_probs  # Confidence score
    })

    output_csv_path = "C://Users//Kieran Taylor//Documents//GitHub//ELEC-292-Group-72//predictions.csv"
    output_df.to_csv(output_csv_path, index=False)
    print(f"Predictions saved to {output_csv_path}")




