# STUDENT's UCO: 505941

# Description:
# This file should be used for performing inference on a network
# Usage: inference.py <dataset_path> <model_path>

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor
from tqdm import tqdm

from network import Generator


def get_image_paths(dataset_path: Path, n: int | None = None):
    if n is not None:
        return list(dataset_path.rglob("*.png"))[:n]
    return list(dataset_path.rglob("*.png"))


# declaration for this function should not be changed
@torch.no_grad()  # do not calculate the gradients
def inference(dataset_path: Path, model_path: Path) -> None:
    """Performs inference on the given dataset using the specified model.

    Args:
        dataset_path: Path to the dataset. The function processes all PNG images in
            this directory (optionally recursively in its subdirectories).
        model_path: Path to the model file.

    Saves:
        predictions to 'output_predictions' folder. The files can be saved in a flat
            structure with the same name as the input file.
    """
    # Check for available GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Computing with {}!".format(device))

    # loading the model
    model = Generator().to(device)
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    model.eval()
    # # From the pix2pix paper:
    # "At inference time, we run the generator net in exactly the same manner as during the training phase.
    # This differs from the usual protocol in that we apply dropout at test time"

    # The outputs seem more colorful when I run inference in the training mode but also some weird artifacts appear (weird blue blobs - perhaps too aggressive dropout?)

    img_paths = get_image_paths(dataset_path, 30)

    for input_img_path in tqdm(img_paths):
        input_img = pil_to_tensor(Image.open(input_img_path)) / 255

        if input_img.shape[0] == 1:
            input_img = input_img.repeat(3, 1, 1)

        generated_image = (
            model(input_img.unsqueeze(0).to(device))[0].permute((1, 2, 0)).cpu().numpy()
        )
        generated_image = (generated_image * 255).astype(np.uint8)

        dir_path = Path("output_predictions")
        dir_path.mkdir(exist_ok=True)
        path_to_save = dir_path / input_img_path.name
        print(f"Saving to: {path_to_save}")
        Image.fromarray(generated_image).save(path_to_save)


# #### code below should not be changed ############################################################################
def main() -> None:
    parser = ArgumentParser(description="Inference script for a neural network.")
    parser.add_argument("dataset_path", type=Path, help="Path to the dataset")
    parser.add_argument("model_path", type=Path, help="Path to the model weights")
    args = parser.parse_args()
    inference(args.dataset_path, args.model_path)


if __name__ == "__main__":
    main()
