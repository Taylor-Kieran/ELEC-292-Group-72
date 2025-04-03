import h5py

with h5py.File("Project/Project/dataset/dataset.hdf5", "r+") as hdf:
    del hdf["segmented/features"]
    