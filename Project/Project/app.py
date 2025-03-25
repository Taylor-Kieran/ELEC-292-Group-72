import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import csv

def open_csv():
    file_path = filedialog.askopenfilename(title="Open CSV File", filetypes=[("CSV files", "*.csv")])
    if file_path:
        display_csv(file_path)

def display_csv(file_path):
    try:
        with open(file_path, 'r', newline='') as file:
            csv_reader = csv.reader(file)
            header = next(csv_reader)
            tree.delete(*tree.get_children())

            tree["columns"] = header
            for col in header:
                tree.heading(col, text=col)
                tree.column(col, width=100)
            for row in csv_reader:
                tree.insert("", "end", values=row)

            status_label.config(text=f"CSV file loaded:{file_path}")
    except Exception as e:
        status_label.config(text=f"Error: {str(e)}")

#parent window
root = tk.Tk()
root.title("Walking Versus Jumping Classifier")

open_button = tk.Button(root, text="Open CSV File", command = open_csv)
open_button.pack(padx=20, pady=10)

tree = ttk.Treeview(root, show="headings")
tree.pack(padx=20, pady=20, fill="both", expand=True)

status_label = tk.Label(root, text="", padx=20, pady=10)
status_label.pack()

root.mainloop()

