# STUDENT's UCO: 505941

# Description:
# This file should contain custom dataset class. The class should subclass the torch.utils.data.Dataset.

from pathlib import Path

from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor


class ImageDataset(Dataset[Tensor]):
    def __init__(self, dataset_path: Path, transform=None, n: int | None = None):
        self.n = n
        self.img_paths = self._get_image_paths(dataset_path)
        self.dataset_path = dataset_path
        self.transform = transform

    def _get_image_paths(self, dataset_path: Path) -> list[str]:
        img_gray_paths = []
        img_rgb_paths = []
        for dir in (dataset_path / "img_gray").iterdir():
            if dir.is_dir():
                for image in dir.iterdir():
                    if image.suffix == ".png":
                        img_gray_paths.append(image.as_posix())

        for dir in (dataset_path / "img_rgb").iterdir():
            if dir.is_dir():
                for image in dir.iterdir():
                    if image.suffix == ".png":
                        img_rgb_paths.append(image.as_posix())
        if self.n is not None:
            assert self.n < len(img_gray_paths), (
                "n must be <= the number of examples in the dataset"
            )
            return list(zip(img_gray_paths, img_rgb_paths))[: self.n]

        return list(zip(img_gray_paths, img_rgb_paths))

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> Tensor:
        img_gray_path, img_rgb_path = self.img_paths[idx]

        img_gray = pil_to_tensor(Image.open(img_gray_path)) / 255
        img_rgb = pil_to_tensor(Image.open(img_rgb_path)) / 255

        if self.transform is not None:
            img_gray = img_gray.permute(1, 2, 0).numpy()
            img_rgb = img_rgb.permute(1, 2, 0).numpy()

            transformed = self.transform(image=img_gray, rgb_image=img_rgb)
            img_gray = transformed["image"]
            img_rgb = transformed["rgb_image"]

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
