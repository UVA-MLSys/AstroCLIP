from datasets import load_dataset, load_from_disk
import os

# This downloads about 60 GB of data
dset = load_dataset(
    'astroclip/data/dataset.py', 
    trust_remote_code=True
)

output_dir = './datasets/astroclip_file'
if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
dset.save_to_disk('./datasets/astroclip_file')

# now you can load the dataset, which is used by the astroclip dataset module
# dset = load_from_disk('./datasets/astroclip_file')