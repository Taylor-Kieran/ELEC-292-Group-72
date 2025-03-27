import h5py
import os
import pandas as pd
from io import StringIO

def display_groups(hdf5_file):
    """Display all groups in the HDF5 file"""
    print("\nAvailable groups:")
    groups = []
    def list_groups(name, obj):
        if isinstance(obj, h5py.Group):
            groups.append(name)
    hdf5_file.visititems(list_groups)
    
    for i, group in enumerate(groups, 1):
        print(f"{i}. {group}")
    return groups

def display_files(group):
    """Display all files in a specific group"""
    print(f"\nFiles in '{group}':")
    files = list(group.keys())
    for i, file in enumerate(files, 1):
        print(f"{i}. {file}")
    return files

def display_csv_content(dataset):
    """Display the content of a CSV dataset"""
    try:
        # Read CSV data from HDF5
        csv_data = dataset[()].decode('utf-8')
        df = pd.read_csv(StringIO(csv_data))
        
        print("\nFile content:")
        print(df.head())  # Show first 5 rows by default
        print(f"\nShape: {df.shape} (rows, columns)")
    except Exception as e:
        print(f"Error displaying file: {e}")

def main():
    # Set up file path
    project_folder = os.path.dirname(os.path.abspath(__file__))
    hdf5_path = os.path.join(project_folder, "dataset", "dataset.hdf5")
    
    if not os.path.exists(hdf5_path):
        print(f"Error: File not found at {hdf5_path}")
        return
    
    with h5py.File(hdf5_path, "r") as hdf5:
        # Step 1: Show all groups
        groups = display_groups(hdf5)
        if not groups:
            print("No groups found in the HDF5 file.")
            return
            
        # Step 2: Let user select a group
        while True:
            try:
                group_choice = int(input("\nEnter group number (0 to exit): "))
                if group_choice == 0:
                    return
                if 1 <= group_choice <= len(groups):
                    selected_group = groups[group_choice-1]
                    break
                print("Invalid choice. Try again.")
            except ValueError:
                print("Please enter a number.")
        
        group = hdf5[selected_group]
        
        # Step 3: Show files in selected group
        files = display_files(group)
        if not files:
            print(f"No files found in '{selected_group}'")
            return
            
        # Step 4: Ask if user wants to view a file
        view_file = input("\nDo you want to view a file? (y/n): ").lower()
        if view_file != 'y':
            return
            
        # Step 5: Let user select a file
        while True:
            try:
                file_choice = int(input("Enter file number (0 to exit): "))
                if file_choice == 0:
                    return
                if 1 <= file_choice <= len(files):
                    selected_file = files[file_choice-1]
                    break
                print("Invalid choice. Try again.")
            except ValueError:
                print("Please enter a number.")
        
        # Step 6: Display the file content
        dataset = group[selected_file]
        if isinstance(dataset, h5py.Dataset):
            if selected_file.endswith('.csv'):
                display_csv_content(dataset)
            else:
                print(f"\nFile '{selected_file}' is not a CSV (first 100 bytes):")
                print(dataset[()][:100])  # Show partial content for non-CSV
        else:
            print("Selected item is not a file.")

if __name__ == "__main__":
    print("HDF5 File Explorer")
    print("-----------------")
    main()