import matplotlib.pyplot as plt
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, recall_score, auc
import h5py
import numpy as np

HDF5_PATH = "Project/Project/dataset/dataset.hdf5"

with h5py.File(HDF5_PATH, "r") as f:
    data = np.array(f["segmented/extracted"])
    columns = [f"feature_{i}" for i in range(data.shape[1] - 1)] + ["label"]  # Change column name
    
    # Convert to DataFrame and ensure the 'label' column is of integer type
    df = pd.DataFrame(data, columns=columns)
    df["label"] = df["label"].astype(int)  



# Extract features
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# assign 10% test 90% train 0% val
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True, random_state=0) 

# Save train and test
with h5py.File(HDF5_PATH, "a") as f:

    # Ensure the "segmented" group exists
    if "segmented" not in f:
        segmented_group = f.create_group("segmented")
    else:
        segmented_group = f["segmented"]

    # Save the train/test datasets
    for name, dataset in [("X_train", X_train),("X_test", X_test), ("y_train", y_train), ("y_test", y_test)]:
        if name in segmented_group:
            del segmented_group[name]
        segmented_group.create_dataset(name, data=dataset)

# Define Standard Scaler
scaler = StandardScaler()

# Define classifier and pipeline
l_reg = LogisticRegression(max_iter=10000)
clf = make_pipeline(StandardScaler(), l_reg)

# Train the model
clf.fit(X_train, y_train)

# Save model path
model_path = "Project/Project/trained_model.pkl"
joblib.dump(clf, model_path)
print(f"Model saved to {model_path}")

# Actaully Predict
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
