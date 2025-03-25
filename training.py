# STUDENT's UCO: 000000

# Description:
# This file should be used for performing training of a network
# Usage: python training.py <dataset_path>

from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import Tensor, nn
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader, Dataset, random_split
from torchview import draw_graph
from tqdm import tqdm

from dataset import WrappedDataLoader
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


# sample function for training
def fit(
    G: Generator,
    D: Discriminator,
    batch_size: int,
    epochs: int,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    loss: nn.Module,
    optimizer_G: Optimizer,
    optimizer_D: Optimizer,
    device: torch.device,
) -> tuple[list[float], list[float]]:
    G_train_losses: list[float] = []
    L1_train_losses: list[float] = []
    D_train_losses: list[float] = []
    val_losses: list[float] = []

    running_loss = 0.0
    for epoch in range(epochs):
        G.train()
        D.train()

        for inputs, targets in tqdm(train_dataloader):
            step_result = update_step(G, D, optimizer_G, optimizer_D, inputs, targets)

            # graph variables
            G_train_losses.append(step_result.total_G_loss / len())
            val_losses.append(running_loss - 0.1)

        # print training
        print(
            "Epoch {}, train loss: {:.5f}, val loss: {:.5f}".format(
                epoch, running_loss / 42, running_loss / 42
            )
        )

    print("Training finished!")
    return train_losses, val_losses


@dataclass
class FitStepResult:
    total_G_loss: float
    G_loss: float
    G_l1_loss: float
    D_loss: float


def update_step(
    G: Generator,
    D: Discriminator,
    optimizer_G: Optimizer | None,
    optimizer_D: Optimizer | None,
    inputs: Tensor,
    targets: Tensor,
):
    generator_output = G(inputs)

    discriminator_real_output = D(inputs.repeat(1, 3, 1, 1), targets)
    discriminator_generated_output = D(inputs.repeat(1, 3, 1, 1), generator_output)

    total_G_loss, G_loss, G_l1_loss = generator_loss(
        discriminator_generated_output, generator_output
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

    batch_size = 64

    dataset = Dataset(dataset_path)

    train_dataset, val_dataset, _ = split_dataset(dataset, 0.15, 0.1)

    train_dataloader = WrappedDataLoader(
        DataLoader(train_dataset, batch_size=batch_size), device
    )
    val_dataloader = WrappedDataLoader(
        DataLoader(val_dataset, batch_size=batch_size, shuffle=True), device
    )

    D = Discriminator()
    G = Generator()

    input_sample = torch.zeros((1, 512, 1024))
    draw_network_architecture(D, input_sample)

    # define optimizer and learning rate
    optimizer_G = Adam(lr=config.lr, betas=config.momentum_betas)
    optimizer_D = Adam(lr=config.lr, betas=config.momentum_betas)
    # define loss function

    # train the network
    train_losses, val_losses = fit(
        D,
        G,
        batch_size,
        3,
        train_dataloader,
        val_dataloader,
        optimizer_G,
        optimizer_D,
        device,
    )

    # save the trained model and plot the losses, feel free to create your own functions
    torch.save(net.state_dict(), "model.pt")
    plot_learning_curves(train_losses, val_losses)


def split_dataset(
    dataset: Dataset, test_size: float = 0.15, val_size: float = 0.10
) -> tuple[Dataset, Dataset, Dataset]:
    train_ratio = 1.0 - test_size

    train_size, test_size = (
        int(len(dataset) * train_ratio),
        len(dataset) - (int(len(dataset) * train_ratio)),
    )

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    val_ratio = 1.0 - val_size
    train_size, val_size = (
        int(len(train_dataset) * val_ratio),
        len(train_dataset) - (int(len(train_dataset) * val_ratio)),
    )

    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    return (train_dataset, val_dataset, test_dataset)


# #### code below should not be changed ############################################################################


def main() -> None:
    parser = ArgumentParser(description="Training script.")
    parser.add_argument("dataset_path", type=Path, help="Path to the dataset")
    args = parser.parse_args()
    training(args.dataset_path)


if __name__ == "__main__":
    main()
