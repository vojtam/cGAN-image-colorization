# STUDENT's UCO: 505941

# Description:
# This file should contain custom dataset class. The class should subclass the torch.utils.data.Dataset.

from pathlib import Path

import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from polars import DataFrame
from sklearn.model_selection import train_test_split
from torch import Tensor
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
        img_gray_path = self.img_paths_df[idx]["img_gray_path"].item()
        img_rgb_path = self.img_paths_df[idx]["img_rgb_path"].item()

        img_gray = pil_to_tensor(Image.open(img_gray_path)) / 255
        img_rgb = pil_to_tensor(Image.open(img_rgb_path)) / 255

        if self.transform is not None:
            img_gray = img_gray.permute(1, 2, 0).numpy()
            img_rgb = img_rgb.permute(1, 2, 0).numpy()

            transformed = self.transform(image=img_gray, rgb_image=img_rgb)
            img_gray = transformed["image"]
            img_rgb = transformed["rgb_image"]
        # img_lab = rgb2lab(img_rgb.permute(1, 2, 0).numpy()).astype(
        #     "float32"
        # )  # Converting RGB to L*a*b
        # img_lab = from_numpy(img_lab).permute(2, 0, 1)
        # L = img_lab[[0], ...] / 50.0 - 1.0  # Between -1 and 1
        # ab = img_lab[[1, 2], ...] / 110.0  # Between -1 and 1

        return img_gray, img_rgb


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
            A.Rotate(limit=30, p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.2),
            A.Normalize(normalization="min_max"),
            ToTensorV2(),
        ],
        additional_targets={"rgb_image": "image"},
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
