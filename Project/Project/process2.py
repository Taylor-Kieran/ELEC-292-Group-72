#this is for processing the overall csv which is walking.csv and jumping.csv

import pandas as pd
import numpy as np

# Load the dataset
file_path = "Project/Project/raw_data/jumping.csv"
df = pd.read_csv(file_path)

"""
# Step 1: Locate indices of NaN values
nan_indices = np.where(pd.isna(dataset))
print("NaN indices:", nan_indices)

# Step 2: Count total NaN values
nan_count = dataset.isna().sum()
print("Total NaN values:", nan_count)

# Step 3: Locate indices of dashes ("-")
dash_indices = np.where(dataset == '-')
print("Dash indices:", dash_indices)

# Step 4: Count total occurrences of dashes
dash_count = (dataset == '-').sum()
print("Total dashes:", dash_count)
"""

import matplotlib.pyplot as plt

# Assuming the CSV has one column of signal values
signal = df.iloc[:, 0].values

# Define function for moving average filter
def moving_average_filter(signal, window_size):
    return np.convolve(signal, np.ones(window_size)/window_size, mode='valid')

# Apply filters with different window sizes
window_sizes = [5, 31, 51]
filtered_signals = {w: moving_average_filter(signal, w) for w in window_sizes}

# Time axis (assuming 100 Hz frequency)
sampling_rate = 100  # Hz
time = np.arange(len(signal)) / sampling_rate

# Plot the original and filtered signals
plt.figure(figsize=(10, 6))
plt.plot(time, signal, label="Original Noisy Signal", alpha=0.5)

for w, filtered in filtered_signals.items():
    time_filtered = time[:len(filtered)]  # Adjust time length for valid mode
    plt.plot(time_filtered, filtered, label=f"Moving Avg (Window={w})")

plt.xlabel("Time (seconds)")
plt.ylabel("Signal Amplitude")
plt.legend()
plt.title("Effect of Moving Average Filter on Noisy Signal")
plt.show()

