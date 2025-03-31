import h5py
import matplotlib.pyplot as plt
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, recall_score, auc
import io
import h5py

HDF5_PATH = "C://Users//Kieran Taylor//Documents//GitHub//ELEC-292-Group-72//Project//Project//dataset//dataset.hdf5"

with h5py.File(HDF5_PATH, "r") as f:
    csv_data = f["segmented//extracted.csv"][()].decode("utf-8")  # Decode bytes to string
    df = pd.read_csv(io.StringIO(csv_data))  # Convert to DataFrame
    print(df.head())

# Extract features (all columns except the label)
X = df.drop(columns=["label"]).values  # Use all columns except "label" as features
y = df["label"].values  # Labels (0 = walking, 1 = jumping)

# assign 10% of the data to test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True, random_state=0)

# Save train and test
with h5py.File(HDF5_PATH, "a") as f:
    # Ensure the "segmented" group exists (or create it if it doesn't)
    if "segmented" not in f:
        segmented_group = f.create_group("segmented")
    else:
        segmented_group = f["segmented"]

    # Save the train/test datasets inside the "segmented" group
    segmented_group.create_dataset("X_train", data=X_train)
    segmented_group.create_dataset("X_test", data=X_test)
    segmented_group.create_dataset("y_train", data=y_train)
    segmented_group.create_dataset("y_test", data=y_test)

# Define Standard Scaler to normalize inputs
scaler = StandardScaler()

# Define classifier and pipeline
l_reg = LogisticRegression(max_iter=10000)
clf = make_pipeline(StandardScaler(), l_reg)

# Train the model
clf.fit(X_train, y_train)

# Save model path
model_path = "C://Users//Kieran Taylor//Documents//GitHub//ELEC-292-Group-72//trained_model.pkl"
joblib.dump(clf, model_path)
print(f"Model saved to {model_path}")

# Get predictions
y_pred = clf.predict(X_test)
y_probs = clf.predict_proba(X_test)[:, 1]  # Probabilities for the positive class (jumping)

# Compute accuracy
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_probs)

print(f"Accuracy: {accuracy:.2f}")
print(f"ROC AUC: {roc_auc:.2f}")

# obtaining the classification recall
recall = recall_score(y_test, y_pred)
print('recall is:', recall)

# plotting confusion matrix
cm = confusion_matrix(y_test, y_pred)
cm_display = ConfusionMatrixDisplay(cm).plot()
plt.show()

# Compute ROC curve and AUC score
fpr, tpr, _ = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)

# Plot ROC Curve
plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})", color="blue")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance (AUC = 0.50)")

# Formatting
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.grid(True)
plt.show()
