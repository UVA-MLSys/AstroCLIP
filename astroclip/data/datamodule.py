from typing import Callable, Dict, List

import datasets
import lightning as L
import torch
from torch import Tensor
from torch.utils.data import default_collate
from torchvision.transforms import CenterCrop

from ..astrodino.data.augmentations import ToRGB


class AstroClipDataloader(L.LightningDataModule):
    def __init__(
        self,
        path: str,
        columns: List[str] = ["image", "spectrum"],
        batch_size: int = 512,
        num_workers: int = 10,
        # persistent_workers: bool = False,
        collate_fn: Callable[[Dict[str, Tensor]], Dict[str, Tensor]] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        # save_hyperparameters converts collate_fn to a dict, which is not callable
        self.collate_fn = collate_fn

    def setup(self, stage: str) -> None:
        # self.dataset = datasets.load_from_disk(self.hparams.path)
        self.dataset = datasets.load_dataset(
            self.hparams.path, cache_dir='datasets/', 
            trust_remote_code=True, keep_in_memory=True
        )
        self.dataset.set_format(type="torch", columns=self.hparams.columns)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset["train"],
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,  # NOTE: disable for debugging
            drop_last=True,
            collate_fn=self.collate_fn,
            pin_memory=True,
            # persistent_workers=self.persistent_workers
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset["test"],
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,  # NOTE: disable for debugging
            drop_last=True,
            collate_fn=self.collate_fn,
            pin_memory=True,
            # persistent_workers=self.persistent_workers
        )


class AstroClipCollator:
    def __init__(
        self,
        center_crop: int = 144,
        bands: List[str] = ["g", "r", "z"],
        m: float = 0.03,
        Q: int = 20,
    ):
        self.center_crop = CenterCrop(center_crop)
        self.to_rgb = ToRGB(bands=bands, m=m, Q=Q)

    def _process_images(self, images):
        # convert to rgb
        img_outs = []
        for img in images:
            rgb_img = torch.tensor(self.to_rgb(img)[None, :, :, :])
            img_outs.append(rgb_img)
        images = torch.concatenate(img_outs)

        images = self.center_crop(images.permute(0, 3, 2, 1))
        return images

    def __call__(self, samples):
        # collate and handle dimensions
        samples = default_collate(samples)
        # process images
        samples["image"] = self._process_images(samples["image"])
        return samples
