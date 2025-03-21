# STUDENT's UCO: 505941

# Description:
# This file should contain custom dataset class. The class should subclass the torch.utils.data.Dataset.

from torch import Tensor
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor

class ImageDataset(Dataset[Tensor]):

    def __init__(self, dataset_path: Path, transform=None):
        self.img_paths = self._get_image_paths(dataset_path)
        self.dataset_path = dataset_path
        self.transform = transform

    def _get_image_paths(self, dataset_path: Path) -> list[str]:
        
        img_paths = []
        for dir in dataset_path.iterdir():
            if dir.is_dir():
                for image in dir.iterdir():
                    if image.suffix == ".png":
                        img_paths.append(image.as_posix()) 
        return img_paths


    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> Tensor:
        img_path = self.img_paths[idx]
        img = pil_to_tensor(Image.open(img_path)) / 255

        if self.transform is not None:
            img = img.permute(1, 2, 0).numpy()
            img = self.transform(image=img)['image']
        return img
