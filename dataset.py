from pathlib import Path

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from polars import DataFrame
from skimage.color import lab2rgb, rgb2lab
from sklearn.model_selection import train_test_split
from torch import Tensor, from_numpy, stack
from torch.utils.data import Dataset
from torchvision.transforms import ColorJitter, ToTensor
from tqdm import tqdm


class ImageDataset(Dataset[Tensor]):
    def __init__(
        self,
        image_paths_df: DataFrame,
        transform=None,
        apply_color_jitter: bool = False,
    ):
        self.img_paths_df = image_paths_df
        self.transform = transform
        self.color_jitter = (
            ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
            if apply_color_jitter
            else None
        )

    def __len__(self) -> int:
        return len(self.img_paths_df)

    def __getitem__(self, idx: int) -> Tensor:
        img_rgb_path = self.img_paths_df[idx]["img_rgb_path"].item()

        img_rgb = Image.open(img_rgb_path).convert("RGB")
        img_rgb_np = np.array(img_rgb)

        if self.transform is not None:
            transformed = self.transform(image=img_rgb_np)
            img_rgb = transformed["image"]
        else:
            img_rgb = ToTensor()(img_rgb)
        # if self.color_jitter is not None:
        #     img_rgb = self.color_jitter(img_rgb)

        L, ab = rgb_to_lab(img_rgb)

        return L, ab


class WrappedDataLoader:
    def __init__(self, dataloader, device):
        self.dataloader = dataloader
        self.device = device

    def __len__(self):
        return len(self.dataloader)

    def __iter__(self):
        batches = iter(self.dataloader)
        for x, y in batches:
            yield x.to(self.device), y.to(self.device)


def get_dataset_stats(ds):
    images = stack([im[1] for im in tqdm(ds)])

    ds_mean = images.mean(dim=(0, 2, 3))
    ds_std = images.std(dim=(0, 2, 3))

    return ds_mean.tolist(), ds_std.tolist()


def get_train_transforms():
    train_transform = A.Compose(
        [
            A.Resize(600, 420),
            A.RandomCrop(512, 384),  # 4x3
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2(),
        ],
        strict=True,
        seed=42,
    )
    return train_transform


def get_val_transforms():
    return A.Compose(
        [
            A.Resize(512, 384),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            A.ToTensorV2(),
        ],
        strict=True,
        seed=42,
    )


def split_dataset(
    img_path_df: DataFrame, test_size: float = 0.15, val_size: float = 0.10
) -> tuple[Dataset, Dataset, Dataset]:
    train_df, test_df = train_test_split(img_path_df, test_size=test_size)
    train_df, valid_df = train_test_split(train_df, test_size=val_size)

    train_dataset = ImageDataset(train_df, get_train_transforms(), True)
    val_dataset = ImageDataset(valid_df, get_val_transforms())
    test_dataset = ImageDataset(test_df, get_val_transforms())

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
    img_gray_paths, img_rgb_paths = sorted(img_gray_paths), sorted(img_rgb_paths)
    if n is not None:
        assert n < len(img_gray_paths), (
            "n must be <= the number of examples in the dataset"
        )
        return DataFrame(
            {"img_gray_path": img_gray_paths[:n], "img_rgb_path": img_rgb_paths[:n]}
        )

    return DataFrame({"img_gray_path": img_gray_paths, "img_rgb_path": img_rgb_paths})


def get_paths_df(dataset_path: Path, extension=".jpg", n: int | None = None):
    if n is not None:
        return DataFrame(
            {"img_rgb_path": list(dataset_path.rglob(f"*{extension}"))[:n]}
        )
    return DataFrame({"img_rgb_path": list(dataset_path.rglob(f"*{extension}"))})


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
