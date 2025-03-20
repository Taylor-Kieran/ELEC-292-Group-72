import os
import pandas as pd
from pandasgui import show
import h5py
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv("C:\\Users\\charl\\.vscode\\290\\ELEC-292-Group-72\\Project\Project\\raw_data\\CharlotteWalking.csv")

acceleration_columns = ['Linear Acceleration x (m/s^2)', 'Linear Acceleration y (m/s^2)', 'Linear Acceleration z (m/s^2)', 'Absolute acceleration (m/s^2)']

dataset = dataset.astype(float)
#skip first row (labels)
time = dataset.iloc[1:, 0]
acceleration = dataset.iloc[1:, 1:]
#choose every 500th element
av_time = time.iloc[::500]
av_acceleration = acceleration.iloc[::500, :]

num_plots = len(acceleration_columns)
fig, ax = plt.subplots(ncols = 1, nrows = num_plots, figsize = ((5*num_plots),5), sharex = True)
for i, col in enumerate(av_acceleration.columns):
    ax[i].plot(av_time, av_acceleration[col], label = acceleration_columns[i], color = 'b', marker='o', linestyle='-')
    ax[i].set_xlabel("Time (s)")
    ax[i].set_ylabel(acceleration_columns[i])
    ax[i].set_title(acceleration_columns[i]+" vs Time")
    ax[i].grid(True)
    ax[i].legend()
fig.tight_layout()
plt.show()

# window_size = 5
moving_avg_dataset = pd.DataFrame()
for column in acceleration_columns:
#  #create a new column for the moving average of each acceleration type
      moving_avg_dataset[f'{column}_Moving_Avg'] = dataset[column].rolling(window=window_size).mean()
#  #show values for both raw and new moving average data
# # gui = show(moving_avg_dataset)