import numpy as np
import pandas as pd
import joblib

file_path = "input"

def predict(file_path):

    # Data Processing

    # Feature Extraction

    # Prediction Model
    # Load the trained model
    model_path = "C://Users//Kieran Taylor//Documents//GitHub//ELEC-292-Group-72//trained_model.pkl"
    clf = joblib.load(model_path)
    print(f"Model loaded from {model_path}")

    # Load the new unlabeled dataset
    unlabeled_test_set = pd.read_csv(file_path)

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

