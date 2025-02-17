# Reproducing

## Installation

## Data

### DESI-LS

DESI-LS Data Release 9 from 2021 January as prepared by Stein et al. (2021b) [1]. Contains 76 million galaxy images.

* [Globus](https://app.globus.org/file-manager?origin_id=59c818dc-8542-46d8-80d9-ab144669c7b6&origin_path=%2Fssl-legacysurvey%2F): The data set is large (20 TB). Contains two subfolders `north` and `south` with 14 and 62 `h5` files respectively. There is also a small toy datasample in the [Github](https://github.com/georgestein/ssl-legacysurvey) at [data/tiny_dataset.h5](https://github.com/georgestein/ssl-legacysurvey/blob/main/data/tiny_dataset.h5).
* Download the data into `datasets/decals` folder.

### DESI Spectra


### Cross-Matched data

The 60GB (80GB after extraction) cross matched data from image and spectra is available to download using the [download_matched_data.py](./scripts/download_matched_data.py) script. This caches the huggingface data and save the dataset to disk at `datasets/astroclip_file` location. You can remove the cached huggingface data afterward. We only need the Dataset saved to disk.

* Run `python scripts/download_matched_data.py`. Takes couple of hours depending on your network speed.

This dataset is used for the multimodal training and all later downstream tasks.

### DESI PROVABGS

They use a sample corresponding to roughly 1 per cent of the DESI Bright Galaxy Survey [5]. From public [DESI homepage](https://data.desi.lbl.gov/doc/access/#globus), we get the [Globus link](https://app.globus.org/file-manager?origin_id=6b4e1f6a-e600-11ed-9b9b-c9bb788c490e) and [direct download link](https://data.desi.lbl.gov/public/edr/vac/edr/provabgs/v1.0/BGS_ANY_full.provabgs.sv3.v0.hdf5). The target file (~3GB) is at location `/edr/vac/edr/provabgs/v1.0/BGS_ANY_full.provabgs.sv3.v0.hdf5`. The POVABGS homepage is [here](https://data.desi.lbl.gov/doc/releases/edr/vac/provabgs/). 

* Download the dataset from [direct download link](https://data.desi.lbl.gov/public/edr/vac/edr/provabgs/v1.0/BGS_ANY_full.provabgs.sv3.v0.hdf5) and save at `/datasets/provabgs/provabgs.hdf5`.

Note, Multimodal Universe has a PROVABGS [dataset](https://huggingface.co/datasets/MultimodalUniverse/desi_provabgs) with 100k rows and 600MB size. Which is smaller, so we are not using.

### Galaxy Zoo DECaLS: Morphology classification

Contains detailed visual morphology measurements from volunteers and deep learning for 314,000 galaxies [2]. [GitHub](ttps://github.com/mwalmsley/zoobot), [Zenodo](https://zenodo.org/record/4573248), [Huggingface](https://huggingface.co/datasets/BigBang/galaxyzoo-decals). 

There is a more updated version of the Galaxy Zoo with different models, called [Galaxy10 DECaLS](https://astronn.readthedocs.io/en/latest/galaxy10.html) [3]. It was also used in the Multimodal Universe paper [4] with different vision models. It comes with 17,736 selected images in 10 broad classes (~2.54GB) with more rigorus filtering. Download the data from [Galaxy10_DECals.h5](https://zenodo.org/records/10845026/files/Galaxy10_DECals.h5).

## Model

Download the pretrained models listed in the readme. Save them in `pretrained` folder. Download the `trained_model/resnet50.ckpt` model from Stein et al. [Globus](https://app.globus.org/file-manager?origin_id=59c818dc-8542-46d8-80d9-ab144669c7b6&origin_path=%2Fssl-legacysurvey%2F) and save it as `stein.ckpt` in the pretrained folder.

## Benchmark

### CLIP Alignment:

Once pretrained, we align the image and spectrum encoder using cross-attention projection heads to maximize the similarity between cross-modal embeddings that correspond to the same galaxy while simultaneously minimizing the similarity between cross-modal embeddings that correspond to different galaxies. Model training can be launched with the following command:
```
spectrum_trainer fit -c configs/astroclip.yaml
```
We train the model using 4 A100 GPUs (on 1 node) for 25k steps or until the validation loss does not increase for a fixed number of steps. This takes roughly 12 hours.

* If `SLURM_NTASKS` not found error, then if running from termila, run `export SLURM_NTASKS=1` or if using slurm set `#SBATCH --ntasks=1` or higher values.

### Classification

* `python downstream_tasks\morphology_classification\morphology_utils\cross_match.py`: This uses the `h5` files in the `datasets/decals` folder and the `galaxy_zoo/gz_decals_volunteers_5.csv` to create a crossmatched `hdf5` file.
* `python .\downstream_tasks\morphology_classification\embed_galaxy_zoo.py`: This will use the previous crossmatched `hdf5` file and models from pretrained folder to save the embeddings.

### Property Estimation

* `python .\downstream_tasks\property_estimation\property_utils\cross_match.py`. For a smaller subset use `python .\downstream_tasks\property_estimation\property_utils\cross_match.py --num_workers 0 --batch_size 64 --max_size 5000`. When on windows, keep the num_workers=0.
* `python .\downstream_tasks\property_estimation\embed_provabgs.py`. Reduce the batch size if running locally. Batch 32 takes ~1h.

For the next 


### Similarity Search

* Run `python .\downstream_tasks\similarity_search\embed_astroclip.py --batch_size 64 --max_size 1024` if you want to reproduce a small subset of the validation data. The smaller batch size helps is run locally. Run `python .\downstream_tasks\similarity_search\embed_astroclip.py` for the originial results.
* Run the `similarity_search.ipynb` notebook.

## References

1. Stein, George, et al. "Self-supervised similarity search for large scientific datasets." arXiv preprint arXiv:2110.13151 (2021).
2. Walmsley, Mike, et al. "Galaxy Zoo DECaLS: Detailed visual morphology measurements from volunteers and deep learning for 314 000 galaxies." Monthly Notices of the Royal Astronomical Society 509.3 (2022): 3966-3988.
3. Leung, W. H., & Bovy, J. (2024). Galaxy10 DECaLS [Data set]. Zenodo. https://doi.org/10.5281/zenodo.10845026
4. Audenaert, Jeroen, et al. "The Multimodal Universe: Enabling Large-Scale Machine Learning with 100TB of Astronomical Scientific Data." arXiv preprint arXiv:2412.02527 (2024).
5. ChangHoon, Hahn, et al. "The desi probabilistic value-added bright galaxy survey (provabgs) mock challenge. The Astrophysical Journal, 945(1):16, March 2023.