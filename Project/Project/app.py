import customtkinter as ctk
import tkinter.ttk as ttk
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import csv
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from predict import predict
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

#main color theme
ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

#main window
app = ctk.CTk()
app.title("Walking vs Jumping Classifier")
app.geometry("800x600")

title_label = ctk.CTkLabel(app, text = "Walking or Jumping", font = ctk.CTkFont(size = 24, weight = "bold"))
title_label.pack(pady=10)

#store CSV data
trained_file = None

#open csv
def open_csv():
    global trained_file
    file_path = filedialog.askopenfilename(title="Open CSV File", filetypes=[("CSV files", "*.csv")])
    if file_path:
        trained_file = predict(file_path)
        classifier_plot(trained_file)
#read and display csv 
'''
def display_csv(file_path):
    global data
    try:
        data = pd.read_csv(file_path)
        tree.delete(*tree.get_children()) #clear any previous rows in treeview
        tree.configure(columns=list(data.columns))
        #for columns
        for col in data.columns:
            tree.heading(col, text=col)
            tree.column(col, width=100)
        for _, row in data.iterrows():
            tree.insert("", "end", values = list(row))
        status_label.configure(text=f"CSV loaded: {file_path}")
    except Exception as e:
        status_label.configure(text=f"Error:{e}")'
'''

#function to generate classifier plot
def classifier_plot(file_path): 
    try:
        df = pd.read_csv(file_path, skiprows = 1, header = None)
        df.columns = ['sample_id', 'predicted_label', 'probability']
        #time is 5 second intervals
        df['time_s'] = df['sample_id']*5
        plt.figure(figsize = (10,4))
        plt.step(df['time_s'], df['predicted_label'], where='post', linewidth=2)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.step(df['time_s'], df['predicted_label'], where='post', linewidth=2)
        #0 is walking, 1 is jumping 
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['Walking', 'Jumping'])
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Activity')
        ax.set_title('Predicted Activity Over Time')
        ax.grid(True)

        canvas = FigureCanvasTkAgg(fig, master=tree_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

        status_label.configure(text="Currently showing Classifier Versus Time Plot")
    except Exception as e:
        status_label.configure(text=f"Error displaying plot {e}")

#function to return csv
def return_csv():
    if trained_file:
        save_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if save_path:
            pd.read_csv(trained_file).to_csv(save_path, index=False)
            status_label.configure(text=f"Classified CSV saved to: {save_path}")
        else:
            status_label.configure(text="Save cancelled")
    else:
        status_label.configure(text="No data to save.")

#button to upload csv
open_button = ctk.CTkButton(app, text="Open CSV File", command=open_csv)
open_button.pack(padx=20, pady=10)

#button to save classified csv
save_button = ctk.CTkButton(app, text="Save Classified CSV File", command = return_csv)
save_button.pack(padx=20, pady=10)

#for displaying the plot
center_frame = ctk.CTkFrame(app)
center_frame.pack(padx=20, pady=20, fill="both", expand=True)
tree_frame = ctk.CTkFrame(center_frame)
tree_frame.pack(fill="both", expand=True, pady=20)

#status messages !
status_label = ctk.CTkLabel(app, text="", anchor="w")
status_label.pack(padx=20, pady=10)

app.mainloop()

