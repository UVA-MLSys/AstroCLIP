# Reproducing

## Installation

## Data

### DESI-LS

DESI-LS Data Release 9 from 2021 January as prepared by Stein et al. (2021b) [1]. Contains 76 million galaxy images.

* [Globus](https://app.globus.org/file-manager?origin_id=59c818dc-8542-46d8-80d9-ab144669c7b6&origin_path=%2Fssl-legacysurvey%2F): The data set is large (20 TB). Contains two subfolders `north` and `south` with 14 and 62 `h5` files respectively. There is also a small toy datasample in the [Github](https://github.com/georgestein/ssl-legacysurvey) at [data/tiny_dataset.h5](https://github.com/georgestein/ssl-legacysurvey/blob/main/data/tiny_dataset.h5).
* Download the data into `datasets/decals` folder.

### Morphology classification

#### Galaxy Zoo DECaLS

Contains detailed visual morphology measurements from volunteers and deep learning for 314,000 galaxies [2]. [GitHub](ttps://github.com/mwalmsley/zoobot), [Zenodo](https://zenodo.org/record/4573248), [Huggingface](https://huggingface.co/datasets/BigBang/galaxyzoo-decals). 

There is a more updated version of the Galaxy Zoo with different models, called [Galaxy10 DECaLS](https://astronn.readthedocs.io/en/latest/galaxy10.html) [3]. It was also used in the Multimodal Universe paper [4] with different vision models. It comes with 17,736 selected images in 10 broad classes (~2.54GB) with more rigorus filtering. Download the data from [Galaxy10_DECals.h5](https://zenodo.org/records/10845026/files/Galaxy10_DECals.h5).

## Model

Download the pretrained models listed in the readme. Save them in `pretrained` folder. Download the `trained_model/resnet50.ckpt` model from Stein et al. [Globus](https://app.globus.org/file-manager?origin_id=59c818dc-8542-46d8-80d9-ab144669c7b6&origin_path=%2Fssl-legacysurvey%2F) and save it as `stein.ckpt` in the pretrained folder.

## Benchmark

### Classification

* `python \downstream_tasks\morphology_classification\morphology_utils\cross_match.py`: This uses the `h5` files in the `datasets/decals` folder and the `galaxy_zoo/gz_decals_volunteers_5.csv` to create a crossmatched `hdf5` file.
* `python .\downstream_tasks\morphology_classification\embed_galaxy_zoo.py`: This will use the previous crossmatched `hdf5` file and models from pretrained folder to save the embeddings.

## References

1. Stein, George, et al. "Self-supervised similarity search for large scientific datasets." arXiv preprint arXiv:2110.13151 (2021).
2. Walmsley, Mike, et al. "Galaxy Zoo DECaLS: Detailed visual morphology measurements from volunteers and deep learning for 314 000 galaxies." Monthly Notices of the Royal Astronomical Society 509.3 (2022): 3966-3988.
3. Leung, W. H., & Bovy, J. (2024). Galaxy10 DECaLS [Data set]. Zenodo. https://doi.org/10.5281/zenodo.10845026
4. Audenaert, Jeroen, et al. "The Multimodal Universe: Enabling Large-Scale Machine Learning with 100TB of Astronomical Scientific Data." arXiv preprint arXiv:2412.02527 (2024).