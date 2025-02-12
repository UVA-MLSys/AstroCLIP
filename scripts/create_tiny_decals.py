import h5py

# Define input and output file paths
input_h5_file = "Datasets/decals/south/images_npix152_000000000_001000000.h5"
output_h5_file = "Datasets/tiny_decals/images.h5"

# Define the maximum number of samples to extract
num_samples = 10000

# Open the input HDF5 file
with h5py.File(input_h5_file, "r") as h5f_in, h5py.File(output_h5_file, "w") as h5f_out:
    # Iterate over all datasets (keys) in the HDF5 file
    for key in h5f_in.keys():
        dataset = h5f_in[key]  # Get the dataset
        
        # Determine the number of available samples (in case it's smaller than num_samples)
        max_samples = min(num_samples, dataset.shape[0])
        
        # Extract the first `max_samples` samples
        data_subset = dataset[:max_samples]

        # Save to new HDF5 file with the same key
        h5f_out.create_dataset(key, data=data_subset, compression="gzip")

        print(f"Saved {max_samples} samples for key: {key}")

print(f"Subset HDF5 file saved at: {output_h5_file}")
