# STUDENT's UCO: 505941

# Description:
# This file should contain custom dataset class. The class should subclass the torch.utils.data.Dataset.

from pathlib import Path

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from polars import DataFrame
from skimage.color import lab2rgb, rgb2lab
from sklearn.model_selection import train_test_split
from torch import Tensor, from_numpy
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor


class ImageDataset(Dataset[Tensor]):
    def __init__(self, image_paths_df: DataFrame, transform=None, n: int | None = None):
        self.n = n
        self.img_paths_df = image_paths_df
        self.transform = transform

    def __len__(self) -> int:
        return len(self.img_paths_df)

    def __getitem__(self, idx: int) -> Tensor:
        # img_gray_path = self.img_paths_df[idx]["img_gray_path"].item()
        img_rgb_path = self.img_paths_df[idx]["img_rgb_path"].item()

        # img_gray = pil_to_tensor(Image.open(img_gray_path))
        img_rgb = pil_to_tensor(Image.open(img_rgb_path))

        if self.transform is not None:
            # img_gray = img_gray.permute(1, 2, 0).numpy()
            img_rgb = img_rgb.permute(1, 2, 0).numpy()

            # transformed = self.transform(image=img_gray, rgb_image=img_rgb)
            transformed = self.transform(image=img_rgb)
            # img_gray = transformed["image"]
            # img_rgb = transformed["rgb_image"]
            img_rgb = transformed["image"]
        # img_gray = img_gray / 255
        img_rgb = img_rgb / 255
        L, ab = rgb_to_lab(img_rgb)
        return L, ab


class WrappedDataLoader:
    def __init__(self, dataloader, device):
        self.dataloader = dataloader
        self.device = device

    def __len__(self):
        return len(self.dataloader)

    def __iter__(self):  # send only one batch to the device every iteration
        batches = iter(self.dataloader)
        for x, y in batches:
            yield x.to(self.device), y.to(self.device)


def get_train_transforms():
    train_transform = A.Compose(
        [
            A.Resize(600, 420),
            A.RandomCrop(512, 384),  # 4x3
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=30.0, p=0.5),
            A.ColorJitter(p=0.6),
            # A.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ToTensorV2(),
        ],
        # additional_targets={"rgb_image": "image"},
    )
    return train_transform


def split_dataset(
    img_path_df: Dataset, test_size: float = 0.15, val_size: float = 0.10
) -> tuple[Dataset, Dataset, Dataset]:
    train_df, test_df = train_test_split(img_path_df, test_size=test_size)
    train_df, valid_df = train_test_split(train_df, test_size=val_size)

    train_dataset = ImageDataset(train_df, get_train_transforms())
    val_dataset = ImageDataset(valid_df)
    test_dataset = ImageDataset(test_df)

    return (train_dataset, val_dataset, test_dataset)


def get_image_paths_df(dataset_path: Path, n: int | None = None) -> list[str]:
    img_gray_paths = []
    img_rgb_paths = []
    for dir in (dataset_path / "img_gray").iterdir():
        if dir.is_dir():
            for image in dir.iterdir():
                if image.suffix in [".png", ".jpg"]:
                    img_gray_paths.append(image.as_posix())

    for dir in (dataset_path / "img_rgb").iterdir():
        if dir.is_dir():
            for image in dir.iterdir():
                if image.suffix in [".png", ".jpg"]:
                    img_rgb_paths.append(image.as_posix())
    if n is not None:
        assert n < len(img_gray_paths), (
            "n must be <= the number of examples in the dataset"
        )
        return DataFrame(
            {"img_gray_path": img_gray_paths[:n], "img_rgb_path": img_rgb_paths[:n]}
        )

    return DataFrame({"img_gray_path": img_gray_paths, "img_rgb_path": img_rgb_paths})


def rgb_to_lab(img_rgb: Tensor):
    img_lab = rgb2lab(img_rgb.permute(1, 2, 0).numpy()).astype("float32")
    img_lab = from_numpy(img_lab).permute(2, 0, 1)
    L = img_lab[[0], ...] / 50.0 - 1.0  # Between -1 and 1
    ab = img_lab[[1, 2], ...] / 110.0  # Between -1 and 1
    return L, ab


def lab_to_rgb_np(L: Tensor, ab: Tensor) -> np.array:
    L = (
        L + 1
    ) * 50.0  # reverse the transformation to range -1 and 1 done in dataset __getitem__
    ab = ab * 110.0
    lab = torch.cat((L, ab), dim=0).permute(1, 2, 0).cpu().detach().numpy()
    rgb_np = lab2rgb(lab)
    return rgb_np


def lab_to_rgb_torch(L: Tensor, ab: Tensor) -> Tensor:
    rgb_torch = torch.from_numpy(lab_to_rgb_batch_np(L, ab)).permute(2, 0, 1)
    return rgb_torch


def lab_to_rgb_batch_np(batch_L: Tensor, batch_ab: Tensor) -> np.array:
    batch_L = (
        batch_L + 1.0
    ) * 50.0  # reverse the transformation to range -1 and 1 done in dataset __getitem__
    batch_ab = batch_ab * 110.0
    batch_Lab = (
        torch.cat((batch_L, batch_ab), dim=1).permute(0, 2, 3, 1).cpu().detach().numpy()
    )
    rgb_imgs = np.stack(
        [lab2rgb(lab_img) for lab_img in batch_Lab], axis=0
    )  # convert and stack them on top of each other

    return rgb_imgs


def lab_to_rgb_batch_torch(batch_L: Tensor, batch_ab: Tensor) -> Tensor:
    rgb_imgs = torch.from_numpy(lab_to_rgb_batch_np(batch_L, batch_ab)).permute(
        0, 3, 1, 2
    )  # convert

    return rgb_imgs
