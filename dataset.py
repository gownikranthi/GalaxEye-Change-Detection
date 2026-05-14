"""
dataset.py
----------
HuggingFace dataloader for the GalaxEye change detection dataset.

Dataset structure (doron333/change-detection-dataset):
  Each split contains 3N samples organised as triplets:
    index i       → POST-event SAR image (grayscale, 1024x1024)
    index i + N   → PRE-event  EO  image (RGB,       1024x1024)
    index i + 2N  → Binary change mask   (0=No-Change, 1=Change)
"""

import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_transforms(split: str, image_size: int, mean: list, std: list):
    """Return albumentations transform pipeline for a given split."""
    if split == "train":
        return A.Compose([
            A.RandomCrop(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.GaussianBlur(p=0.2),
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.CenterCrop(image_size, image_size),
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ])


class ChangeDetectionDataset(Dataset):
    """
    PyTorch Dataset for EO-SAR binary change detection.

    Parameters
    ----------
    hf_dataset : HuggingFace Dataset split (e.g. ds['train'])
    split      : 'train', 'val', or 'test'
    image_size : spatial size of output patches (default 256)
    """

    MEAN = [0.485, 0.456, 0.406, 0.5]   # RGB channels + SAR channel
    STD  = [0.229, 0.224, 0.225, 0.5]

    def __init__(self, hf_dataset, split: str = "train",
                 image_size: int = 256):
        self.ds         = hf_dataset
        self.split      = split
        self.image_size = image_size
        self.N          = len(hf_dataset) // 3   # number of triplets
        self.transform  = get_transforms(split, image_size,
                                         self.MEAN, self.STD)

    def __len__(self) -> int:
        return self.N

    def __getitem__(self, idx: int):
        # ── Load raw arrays ──────────────────────────────────────
        post_sar = np.array(self.ds[idx]["image"])               # SAR
        pre_eo   = np.array(self.ds[self.N + idx]["image"])      # EO
        mask     = np.array(self.ds[self.N * 2 + idx]["image"])  # mask

        # ── Ensure correct shapes ────────────────────────────────
        if post_sar.ndim == 2:
            post_sar = post_sar[:, :, np.newaxis]          # (H,W) → (H,W,1)

        if pre_eo.ndim == 2:
            pre_eo = np.stack([pre_eo] * 3, axis=-1)       # grey → RGB

        # ── Early fusion: EO(3ch) + SAR(1ch) = 4ch ──────────────
        image_4ch = np.concatenate(
            [pre_eo, post_sar], axis=-1
        ).astype(np.float32)

        mask = mask.astype(np.float32)

        # ── Augment / normalise ──────────────────────────────────
        out   = self.transform(image=image_4ch, mask=mask)
        image = out["image"]          # (4, H, W) float tensor
        mask  = out["mask"]           # (H, W)    float tensor

        return image, mask.unsqueeze(0)   # mask → (1, H, W)

    def get_raw_triplet(self, idx: int):
        """Return raw (unnormalised) numpy arrays for visualisation."""
        post_sar = np.array(self.ds[idx]["image"])
        pre_eo   = np.array(self.ds[self.N + idx]["image"])
        mask     = np.array(self.ds[self.N * 2 + idx]["image"])
        return pre_eo, post_sar, mask
