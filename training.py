# STUDENT's UCO: 000000

# Description:
# This file should be used for performing training of a network
# Usage: python training.py <dataset_path>

import time
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import Tensor, nn
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader
from torchview import draw_graph
from tqdm import tqdm

from dataset import WrappedDataLoader, get_image_paths_df, split_dataset
from network import Discriminator, Generator, discriminator_loss, generator_loss


@dataclass
class config:
    lr: float = 0.0002
    momentum_betas: tuple[float, float] = (0.5, 0.999)


# sample function for model architecture visualization
# draw_graph function saves an additional file: Graphviz DOT graph file, it's not necessary to delete it
def draw_network_architecture(net: nn.Module, input_sample: Tensor) -> None:
    # saves visualization of model architecture to the model_architecture.png
    draw_graph(
        net,
        input_sample,
        graph_dir="TB",
        save_graph=True,
        filename="model_architecture",
        expand_nested=True,
    )


# sample function for losses visualization
def plot_learning_curves(
    train_losses: list[float], validation_losses: list[float]
) -> None:
    plt.figure(figsize=(10, 5))
    plt.title("Train and Evaluation Losses During Training")
    plt.plot(train_losses, label="train_loss")
    plt.plot(validation_losses, label="validation_loss")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("learning_curves.png")


@dataclass
class FitStepResult:
    total_G_loss: float
    G_loss: float
    G_l1_loss: float
    D_loss: float


def generate_image(
    model: Generator, input: Tensor, target: Tensor, epoch: int, plot_subtitle: str = ""
):
    # input has a shape C x H x W
    # target has a shape C x H x W
    predicted = model(input.unsqueeze(0))  # add a batch dimension
    plt.figure(figsize=(15, 15))

    input_np = input.permute(1, 2, 0).cpu().detach().numpy()
    target_np = target.permute(1, 2, 0).cpu().detach().numpy()
    predicted_np = predicted.squeeze().permute(1, 2, 0).cpu().detach().numpy()

    display_list = [input_np, target_np, predicted_np]
    title = ["Input Image", "Ground Truth", "Predicted Image"]

    for i in range(3):
        plt.subplot(1, 3, i + 1)

        plt.title(title[i])
        if title[i] == "Input Image":
            plt.imshow(display_list[i] * 0.5 + 0.5, cmap="gray")
        else:
            plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis("off")
    plt.suptitle(plot_subtitle)
    path = f"gen_images/generated_image_{epoch}.png"
    print(f"Sample colorized image saved to {path}")
    plt.savefig(path)


# sample function for training
def fit(
    G: Generator,
    D: Discriminator,
    epochs: int,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    optimizer_G: Optimizer,
    optimizer_D: Optimizer,
    input_to_vizu: Tensor,
    target_to_vizu: Tensor,
    device: torch.device,
) -> tuple[list[float], list[float]]:
    G_train_losses: list[float] = []
    L1_train_losses: list[float] = []
    D_train_losses: list[float] = []

    G_val_losses: list[float] = []
    L1_val_losses: list[float] = []
    D_val_losses: list[float] = []

    for epoch in range(epochs):
        G.train()
        D.train()
        torch.cuda.synchronize()
        epoch_start_time = time.time()

        train_G_loss = 0.0
        train_D_loss = 0.0
        train_L1_loss = 0.0

        for inputs, targets in tqdm(train_dataloader):
            step_result = fit_step(G, D, optimizer_G, optimizer_D, inputs, targets)

            train_G_loss += step_result.total_G_loss.item()
            train_D_loss += step_result.D_loss.item()
            train_L1_loss += step_result.G_l1_loss.item()

        n_train_batches = len(train_dataloader)
        G_train_losses.append(train_G_loss / n_train_batches)
        D_train_losses.append(train_D_loss / n_train_batches)
        L1_train_losses.append(train_L1_loss / n_train_batches)

        G.eval()
        D.eval()

        val_G_loss = 0.0
        val_D_loss = 0.0
        val_L1_loss = 0.0
        n_val_batches = len(val_dataloader)

        with torch.no_grad():
            for inputs, targets in tqdm(val_dataloader):
                step_result = fit_step(
                    G, D, None, None, inputs, targets
                )  # optimizers are none -> don't update

                val_G_loss += step_result.total_G_loss.item()
                val_D_loss += step_result.D_loss.item()
                val_L1_loss += step_result.G_l1_loss.item()

        G_val_losses.append(val_G_loss / n_val_batches)
        D_val_losses.append(val_D_loss / n_val_batches)
        L1_val_losses.append(val_L1_loss / n_val_batches)

        torch.cuda.synchronize()
        epoch_end_time = time.time()

        print(
            f"Epoch: {epoch}, time_elapsed: {epoch_end_time - epoch_start_time:.2f} seconds | ",
            f"train_G_loss: {train_G_loss:.3f} | train_L1_loss: {train_L1_loss:.3f} | train_D_loss: {train_D_loss:.3f} | ",
            f"val_G_loss: {val_G_loss:.3f} | val_L1_loss: {val_L1_loss:.3f} | val_D_loss: {val_D_loss:.3f}",
            sep="\n",
        )

        generate_image(G, input_to_vizu, target_to_vizu, epoch, f"epoch: {epoch}")
    print("Training finished!")
    return G_train_losses, D_train_losses


def fit_step(
    G: Generator,
    D: Discriminator,
    optimizer_G: Optimizer | None,
    optimizer_D: Optimizer | None,
    inputs: Tensor,
    targets: Tensor,
):
    generator_output = G(inputs)

    inputs = inputs.repeat(1, 3, 1, 1)

    discriminator_real_output = D(inputs, targets)
    discriminator_generated_output = D(inputs, generator_output)

    total_G_loss, G_loss, G_l1_loss = generator_loss(
        discriminator_generated_output, generator_output, targets
    )
    D_loss = discriminator_loss(
        discriminator_real_output, discriminator_generated_output
    )

    if optimizer_G is not None:
        optimizer_G.zero_grad()
        total_G_loss.backward()
        optimizer_G.step()

    if optimizer_D is not None:
        optimizer_D.zero_grad()
        D_loss.backward()
        optimizer_D.step()

    return FitStepResult(total_G_loss, G_loss, G_l1_loss, D_loss)


# declaration for this function should not be changed
def training(dataset_path: Path) -> None:
    """Performs training on the given dataset.

    Args:
        dataset_path: Path to the dataset.

    Saves:
        - model.pt (trained model)
        - learning_curves.png (learning curves generated during training)
        - model_architecture.png (a scheme of model's architecture)
    """
    # Check for available GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Computing with {}!".format(device))

    batch_size = 128

    img_paths_df = get_image_paths_df(dataset_path)

    train_dataset, val_dataset, _ = split_dataset(img_paths_df, 0.15, 0.1)

    train_dataloader = WrappedDataLoader(
        DataLoader(train_dataset, batch_size=batch_size), device
    )
    val_dataloader = WrappedDataLoader(
        DataLoader(val_dataset, batch_size=batch_size, shuffle=True), device
    )

    torch.set_float32_matmul_precision("high")

    D = Discriminator().to(device)
    G = Generator().to(device)

    D, G = (
        torch.compile(D, mode="reduce-overhead"),
        torch.compile(G, mode="reduce-overhead"),
    )

    # define optimizer and learning rate
    optimizer_G = Adam(G.parameters(), lr=config.lr, betas=config.momentum_betas)
    optimizer_D = Adam(D.parameters(), lr=config.lr, betas=config.momentum_betas)
    # define loss function

    # train the network
    train_losses, val_losses = fit(
        G,
        D,
        3,
        train_dataloader,
        val_dataloader,
        optimizer_G,
        optimizer_D,
        val_dataset[0][0].to(device),
        val_dataset[0][1].to(device),
        device,
    )


# #### code below should not be changed ############################################################################


def main() -> None:
    parser = ArgumentParser(description="Training script.")
    parser.add_argument("dataset_path", type=Path, help="Path to the dataset")
    args = parser.parse_args()
    training(args.dataset_path)


if __name__ == "__main__":
    main()
