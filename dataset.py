from pathlib import Path

import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from polars import DataFrame
from sklearn.model_selection import train_test_split
from torch import Tensor, stack
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
        # img_gray_path = self.img_paths_df[idx]["img_gray_path"].item()
        img_rgb_path = self.img_paths_df[idx]["img_rgb_path"].item()

        img_rgb = Image.open(img_rgb_path).convert("RGB")
        img_gray = img_rgb.convert("L")
        img_rgb = ToTensor()(img_rgb)
        img_gray = ToTensor()(img_gray).repeat(
            3, 1, 1
        )  # convert to 3-channel grayscale

        if self.transform is not None:
            img_gray = img_gray.permute(1, 2, 0).numpy()
            img_rgb = img_rgb.permute(1, 2, 0).numpy()

            transformed = self.transform(image=img_gray, rgb_image=img_rgb)
            img_gray = transformed["image"]
            img_rgb = transformed["rgb_image"]

        if self.color_jitter is not None:
            img_rgb = self.color_jitter(img_rgb)
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
            ToTensorV2(),
        ],
        additional_targets={"rgb_image": "image"},
        strict=True,
        seed=42,
    )
    return train_transform


def get_val_transforms():
    return A.Compose(
        [
            A.Resize(512, 384),
            A.ToTensorV2(),
        ],
        additional_targets={"rgb_image": "image"},
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
