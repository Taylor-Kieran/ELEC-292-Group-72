

#this is for finding out the groups present in dataset hdf5
"""import h5py
import os

# Define the HDF5 file path
project_folder = os.path.dirname(os.path.abspath(__file__))  # Get script's directory
hdf5_path = os.path.join(project_folder, "dataset", "dataset.hdf5")  # Path to dataset.hdf5

# Open the HDF5 file and list all groups
with h5py.File(hdf5_path, "r") as hdf5:
    def list_groups(name, obj):
        if isinstance(obj, h5py.Group):
            print(name)

    hdf5.visititems(list_groups)"""



#this if for finding out the files present in raw group of the hdf5
import h5py
import os

# Define the HDF5 file path
project_folder = os.path.dirname(os.path.abspath(__file__))  # Get script's directory
hdf5_path = os.path.join(project_folder, "dataset", "dataset.hdf5")  # Path to dataset.hdf5

# Open the HDF5 file and list all CSV file names in "raw"
with h5py.File(hdf5_path, "r") as hdf5:
    if "raw" in hdf5:
        for file in hdf5["raw"].keys():  # Directly iterate over datasets in "raw"
            print(file)
    else:
        print("'raw' group not found in the HDF5 file.")

        
#this if for finding out the files present in pre_processed group of the hdf5
"""import h5py
import os

# Define the HDF5 file path
project_folder = os.path.dirname(os.path.abspath(__file__))  # Get script's directory
hdf5_path = os.path.join(project_folder, "dataset", "dataset.hdf5")  # Path to dataset.hdf5

# Open the HDF5 file and check contents of pre_processed
with h5py.File(hdf5_path, "r") as hdf5:
    if "pre_processed" in hdf5:
        contents = list(hdf5["pre_processed"].keys())
        if contents:
            print("\n".join(contents))  # Print each item in pre_processed on a new line
        else:
            print("pre_processed is empty")
    else:
        print("pre_processed group does not exist")"""

        




        
      
