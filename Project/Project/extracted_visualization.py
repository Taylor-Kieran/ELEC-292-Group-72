import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Load the dataset from HDF5 file
hdf5_path = "Project/Project/dataset/dataset.hdf5"  # Update if needed
with h5py.File(hdf5_path, "r") as f:
    features = np.array(f["segmented/extracted"])  # Adjust path if necessary

# Separate features and labels
X = features[:, :-1]  # All columns except the last one
y = features[:, -1]   # Last column contains labels (0 = Walking, 1 = Jumping)

# Apply t-SNE for dimensionality reduction
tsne = TSNE(n_components=2, perplexity=30, random_state=42)  # Adjust perplexity if needed
X_tsne = tsne.fit_transform(X)

# Scatter plot of t-SNE transformed data
plt.figure(figsize=(8, 6))
plt.scatter(X_tsne[y == 0, 0], X_tsne[y == 0, 1], label="Walking", color="blue", alpha=0.6, s=30)
plt.scatter(X_tsne[y == 1, 0], X_tsne[y == 1, 1], label="Jumping", color="red", alpha=0.6, s=30)
plt.legend()
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.title("t-SNE Visualization of Extracted Features")
plt.grid(True)
plt.show()
