import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import csv
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def open_csv():
    file_path = filedialog.askopenfilename(title="Open CSV File", filetypes=[("CSV files", "*.csv")])
    if file_path:
        display_csv(file_path)

#display the opened CSV
def display_csv(file_path):
    try:
        global data
        data = pd.read_csv(file_path)
        tree.delete(*tree.get_children())
        
        # Set up columns
        tree["columns"] = list(data.columns)
        for col in data.columns:
            tree.heading(col, text=col)
            tree.column(col, width=100)

        # Insert rows into the treeview
        for row in data.values.tolist():
            tree.insert("", "end", values=row)

        status_label.config(text=f"CSV file loaded:{file_path}")
    except Exception as e:
        status_label.config(text=f"Error: {str(e)}")

#display plot, this will be replaced with our code and actual plots...!
def generate_plot():
    fig, ax = plt.subplots()
    x,y = np.random.rand(2,6)
    z = np.array([0,0,1,1,0,1])
    ax.scatter(x,y,marker='v', c=z, cmap="nipy_spectral")
    plt.show()
    status_label.config(text="Plot generated")

#option to save the new trained data as csv, this needs to be added onto as well
def return_csv():
    if data is not None:
        save_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files","*.csv")])
        if save_path:
            data.to_csv(save_path, index = False)
            status_label.config(text=f"CSV saved to: {save_path}")
        else:
            status_label.config(text="No data to save")

#parent window
root = tk.Tk()
root.title("Walking Versus Jumping Classifier")

#open csv button
open_button = tk.Button(root, text="Open CSV File", command = open_csv)
open_button.pack(padx=20, pady=10)

#dropdown menu
options = ["Generate Plot", "Return CSV"]
dropdown = ttk.Combobox(root, values=options, state="readonly")
dropdown.set("Select an Option")
dropdown.pack(padx=20, pady=10)

# Action button to trigger the chosen option
def on_select():
    choice = dropdown.get()
    if choice == "Generate Plot":
        generate_plot()
    elif choice == "Return CSV":
        return_csv()
action_button = tk.Button(root, text="Execute Action", command=on_select)
action_button.pack(padx=20, pady=10)

tree = ttk.Treeview(root, show="headings")
tree.pack(padx=20, pady=20, fill="both", expand=True)

status_label = tk.Label(root, text="", padx=20, pady=10)
status_label.pack()

root.mainloop()

