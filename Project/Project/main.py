import numpy as np
import pandas as pd
import h5py
from process import process_hdf5
from feature_extraction import feature_extraction, normalize_features


file_path = "Project/Project/dataset/dataset.hdf5"

#processing
processed_data = process_hdf5(file_path)

with h5py.File(file_path, "r+") as f:
    preprocessed_group = f["pre-processed"]
    for name, df in processed_data.items():
        if name in preprocessed_group:
            del preprocessed_group[name]  
        preprocessed_group.create_dataset(name, data=df.to_numpy())

print("Pre-processing complete. Data saved in 'pre-processed' group.")

#feature extracting
extracted_df = feature_extraction(file_path)

#normalization
normalized_df = normalize_features(extracted_df)
normalized_df["label"] = normalized_df["label"].astype(int)

with h5py.File(file_path, "r+") as f:
    if "segmented" not in f:
        segmented_group = f.create_group("segmented")
    else:
        segmented_group = f["segmented"]

    if "extracted" in segmented_group:
        del segmented_group["extracted"]

    segmented_group.create_dataset("extracted", data=normalized_df.to_numpy())

print("Feature extraction and normalization complete. Data saved in 'segmented' group.")

