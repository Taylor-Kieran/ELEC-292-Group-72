import numpy as np
import pandas as pd
import joblib
import h5py
from process import process_dataset
from feature_extraction import normalize_features, extract_features_from_segment

csv_path = "Project/Project/dataP.csv"
MODEL_PATH = "Project/Project/trained_model.pkl"
OUTPUT_CSV_PATH = "Project/Project/pred.csv"

import numpy as np
import pandas as pd

def feature_extraction(df):
    extracted_features = []
    
    # Converting dataframe to numpy array for easier processing
    data = df.to_numpy()
    
    # Segmenting the data into 5 second windows for feature extraction
    num_segments = len(data) // 500 
    for i in range(num_segments):
        segment = data[i * 500:(i + 1) * 500, :]
        
        new_row = extract_features_from_segment(segment)

        extracted_features.append(new_row)
    
    #
    return pd.DataFrame(extracted_features)



def predict(file_path):
    print(f"[DEBUG] Processing HDF5 file: {file_path}")
    
    # reading the data
    df = pd.read_csv(file_path)
    # processing
    processed_data = process_dataset(df)
    #feature extraction and normalization
    extracted_df = feature_extraction(processed_data)
    normalized_df = normalize_features(extracted_df)
  
    # Load trained model
    clf = joblib.load(MODEL_PATH)
    print(f"[DEBUG] Model loaded from {MODEL_PATH}")

    # to ensure data matches expected format
    X_unlabeled = normalized_df.drop(columns=["label"], errors="ignore").values

    # Applying the same feature scaling
    if hasattr(clf, "named_steps") and "standardscaler" in clf.named_steps:
        X_unlabeled = clf.named_steps["standardscaler"].transform(X_unlabeled)
    else:
        print("[WARNING] Model does not contain 'standardscaler'. Proceeding without scaling.")

    # Making predictions
    predicted_labels = clf.predict(X_unlabeled)
    predicted_probs = clf.predict_proba(X_unlabeled)[:, 1]  

    # Saving results to CSV
    output_df = pd.DataFrame({
        "Sample_ID": np.arange(len(predicted_labels)),
        "Predicted_Label": predicted_labels, 
        "Jumping_Probability": predicted_probs
    })
    output_df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"[INFO] Predictions saved to {OUTPUT_CSV_PATH}")

    return OUTPUT_CSV_PATH


