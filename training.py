# Usage: python training.py <dataset_path>

import random
import time
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader
from torchview import draw_graph
from tqdm import tqdm

from dataset import (
    WrappedDataLoader,
    get_paths_df,
    lab_to_rgb_batch_np,
    lab_to_rgb_np,
    split_dataset,
)
from evaluation import dssim
from network import GAN, Discriminator, Generator, discriminator_loss, generator_loss


def set_global_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


BCELogitsLoss = nn.BCEWithLogitsLoss()
L1Loss = torch.nn.L1Loss()


@dataclass
class config:
    G_lr: float = 0.0002
    D_lr: float = 0.00001
    batch_size: int = 16
    LAMBDA: int = 65
    momentum_betas: tuple[float, float] = (0.5, 0.999)
    epoch_num: int = 600
    random_seed: int = 42


set_global_random_seed(config.random_seed)


#  function for model architecture visualization
# draw_graph function saves an additional file: Graphviz DOT graph file
def draw_network_architecture(
    net: nn.Module, input_sample: Tensor, filename: str = "model_architecture"
) -> None:
    # saves visualization of model architecture to the model_architecture.png
    draw_graph(
        net,
        input_sample,
        graph_dir="TB",
        save_graph=True,
        filename=filename,
        expand_nested=True,
    )


def plot_learning_curves(losses_dict: dict[list[float]]) -> None:
    plt.figure(figsize=(10, 5))
    plt.title("Train and Evaluation Losses During Training")
    for label, loss in losses_dict.items():
        plt.plot(loss, label=label)
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
    dssim: float = 0.0


def scale_to_zero_one(image_np):
    return (image_np - image_np.min()) / (image_np.max() - image_np.min())


@torch.no_grad
def generate_image(
    model: Generator,
    input: Tensor,
    target: Tensor,
    epoch: int,
    plot_subtitle: str = "",
    save: bool = True,
    save_path=Path("gen_images/single"),
):
    # input has a shape C x H x W
    # target has a shape C x H x W
    model.eval()
    predicted = model(
        input.unsqueeze(0)
    )  # add a batch dimension so that I can feed it to the Generator
    model.train()
    predicted_np = lab_to_rgb_np(input, predicted.squeeze())
    target_np = lab_to_rgb_np(input, target)
    input_np = (input.permute(1, 2, 0).cpu().detach().numpy() + 1.0) / 2.0

    predicted_np = np.clip(predicted_np, 0.0, 1.0)
    target_np = np.clip(target_np, 0.0, 1.0)
    input_np = np.clip(input_np, 0.0, 1.0)

    plt.figure(figsize=(10, 5))

    display_list = [input_np, target_np, predicted_np]
    title = ["Input Image", "Ground Truth", "Predicted Image"]

    for i in range(3):
        plt.subplot(1, 3, i + 1)

        plt.title(title[i])
        if title[i] == "Input Image":
            plt.imshow(display_list[i], cmap="gray")
        else:
            plt.imshow(display_list[i])
        plt.axis("off")
    plt.suptitle(plot_subtitle)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if save:
        save_path.mkdir(exist_ok=True)
        path = save_path / f"generated_image_{epoch}.png"
        print(f"Sample colorized image saved to {path}")
        plt.savefig(path)
        plt.close()
    else:
        plt.show()


@torch.no_grad
def visualize_batch(
    model: Generator,
    inputs: Tensor,  # Batch of L channels
    targets: Tensor,  # Batch of ab channels
    epoch: int,
    num_images: int = 4,
    save=True,
    save_path=Path("gen_images/batch"),
):
    model.eval()
    num_images = min(num_images, inputs.shape[0])

    inputs_slice = inputs[:num_images]
    targets_slice = targets[:num_images]
    predicted_ab_slice = model(inputs_slice)
    model.train()

    generated_imgs = lab_to_rgb_batch_np(inputs_slice, predicted_ab_slice)
    ground_truth = lab_to_rgb_batch_np(inputs_slice, targets_slice)

    grayscale = (inputs_slice + 1.0) / 2.0  # Map from [-1, 1] to [0, 1]
    grayscale = grayscale.repeat(1, 3, 1, 1).cpu().permute(0, 2, 3, 1).numpy()

    generated_imgs = np.clip(generated_imgs, 0.0, 1.0)
    ground_truth = np.clip(ground_truth, 0.0, 1.0)
    grayscale = np.clip(grayscale, 0.0, 1.0)

    fig, axs = plt.subplots(3, num_images, figsize=(num_images * 2.5, 7.5))
    row_titles = ["Grayscale", "Ground Truth", "Generated"]

    for i in range(num_images):
        axs[0, i].imshow(grayscale[i])
        axs[1, i].imshow(ground_truth[i])
        axs[2, i].imshow(generated_imgs[i])

    for i, ax_row in enumerate(axs):
        for j, ax in enumerate(ax_row):
            ax.axis("off")
            if j == 0:
                ax.set_title(row_titles[i], fontsize=12, loc="left", pad=10)

    fig.suptitle(f"Epoch: {epoch}", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if save:
        save_path.mkdir(exist_ok=True)
        path = save_path / f"generated_images_{epoch}.png"
        print(f"Sample colorized batch saved to {path}")
        plt.savefig(path)
        plt.close()
    else:
        plt.show()


def train_step(
    G: Generator,
    D: Discriminator,
    optimizer_G: Optimizer | None,
    optimizer_D: Optimizer | None,
    inputs: Tensor,
    targets: Tensor,
    device: torch.device,
):
    optimizer_D.zero_grad()
    optimizer_G.zero_grad()

    generator_output = G(inputs)
    generated = torch.concat(
        (inputs, generator_output.detach()), dim=1
    )  # Note to self: need to use detach here to keep the computational graphs separate -> otherwise backprop would break
    real = torch.concat((inputs, targets), dim=1)
    discriminator_generated_output = D(generated)
    discriminator_real_output = D(real)

    D_loss = discriminator_loss(
        discriminator_generated_output, discriminator_real_output, smoothing_factor=0.9
    )
    D_loss.backward()
    optimizer_D.step()

    G_fake = torch.concat((inputs, generator_output), dim=1)
    D_fake_output = D(G_fake)
    G_loss, G_L1_loss = generator_loss(
        D_fake_output, generator_output, targets, LAMBDA=config.LAMBDA
    )
    G_total_loss = G_loss + G_L1_loss

    G_total_loss.backward()
    optimizer_G.step()

    return FitStepResult(G_total_loss, G_loss, G_L1_loss, D_loss)


@torch.no_grad
def valid_step(
    G: Generator,
    D: Discriminator,
    inputs: Tensor,
    targets: Tensor,
    device: torch.device,
    epoch: int,
):
    generator_output = G(inputs)

    generated = torch.cat((inputs, generator_output), dim=1)
    real = torch.cat((inputs, targets), dim=1)
    discriminator_generated_output = D(generated)
    discriminator_real_output = D(real)

    D_loss = discriminator_loss(
        discriminator_generated_output, discriminator_real_output
    )

    G_loss, G_L1_loss = generator_loss(
        discriminator_generated_output,
        generator_output,
        targets,
        LAMBDA=config.LAMBDA,
    )

    G_total_loss = G_loss + G_L1_loss

    if epoch % 20 == 0:
        pred_RGB = lab_to_rgb_batch_np(inputs, generator_output) * 255
        ref_RGB = lab_to_rgb_batch_np(inputs, targets) * 255
        dssim_score = 0
        for ref_img, pred_img in zip(ref_RGB, pred_RGB):
            dssim_score += dssim(ref_img.astype(np.uint8), pred_img.astype(np.uint8))
        dssim_score /= inputs.shape[0]
        return FitStepResult(G_total_loss, G_loss, G_L1_loss, D_loss, dssim=dssim_score)

    return FitStepResult(G_total_loss, G_loss, G_L1_loss, D_loss)


# sample function for training
def fit(
    G: Generator,
    D: Discriminator,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    optimizer_G: Optimizer,
    optimizer_D: Optimizer,
    device: torch.device,
) -> tuple[list[float], list[float]]:
    G_train_losses: list[float] = []
    L1_train_losses: list[float] = []
    D_train_losses: list[float] = []

    G_val_losses: list[float] = []
    L1_val_losses: list[float] = []
    D_val_losses: list[float] = []

    for epoch in range(config.epoch_num):
        G.train()
        D.train()
        torch.cuda.synchronize()
        epoch_start_time = time.time()

        train_G_loss = 0.0
        train_D_loss = 0.0
        train_L1_loss = 0.0

        n_batches = 0
        for inputs, targets in tqdm(train_dataloader):
            n_batches += 1
            step_result = train_step(
                G,
                D,
                optimizer_G,
                optimizer_D,
                inputs,
                targets,
                device,
            )

            train_G_loss += step_result.G_loss.item()
            train_D_loss += step_result.D_loss.item()
            train_L1_loss += step_result.G_l1_loss.item()

        G_train_losses.append(train_G_loss / n_batches)
        D_train_losses.append(train_D_loss / n_batches)
        L1_train_losses.append(train_L1_loss / n_batches)

        # VALIDATION
        G.eval()
        D.eval()

        val_G_loss = 0.0
        val_D_loss = 0.0
        val_L1_loss = 0.0
        dssim = 0.0
        n_batches = 0
        for inputs, targets in tqdm(val_dataloader):
            n_batches += 1
            step_result = valid_step(
                G,
                D,
                inputs,
                targets,
                device,
                epoch,
            )

            val_G_loss += step_result.G_loss.item()
            val_D_loss += step_result.D_loss.item()
            val_L1_loss += step_result.G_l1_loss.item()
            dssim += step_result.dssim

        G_val_losses.append(val_G_loss / n_batches)
        D_val_losses.append(val_D_loss / n_batches)
        L1_val_losses.append(val_L1_loss / n_batches)

        torch.cuda.synchronize()
        epoch_end_time = time.time()

        mlflow.log_metric("train_G_loss", train_G_loss, step=epoch)
        mlflow.log_metric("train_D_loss", train_D_loss, step=epoch)
        mlflow.log_metric("train_L1_loss", train_L1_loss, step=epoch)
        mlflow.log_metric("val_G_loss", val_G_loss, step=epoch)
        mlflow.log_metric("val_D_loss", val_D_loss, step=epoch)
        mlflow.log_metric("val_L1_loss", val_L1_loss, step=epoch)
        mlflow.log_metric(
            "epoch_duration", epoch_end_time - epoch_start_time, step=epoch
        )
        print(
            f"Epoch: {epoch}, time_elapsed: {epoch_end_time - epoch_start_time:.2f} seconds | ",
            f"train_G_loss: {G_train_losses[-1]:.3f} | train_L1_loss: {L1_train_losses[-1]:.3f} | train_D_loss: {D_train_losses[-1]:.3f} | ",
            f"val_G_loss: {G_val_losses[-1]:.3f} | val_L1_loss: {L1_val_losses[-1]:.3f} | val_D_loss: {D_val_losses[-1]:.3f}",
            sep="\n",
        )
        if epoch % 20 == 0:
            avg_dssim = dssim / n_batches
            print(f"average dssim score : {avg_dssim}")
            mlflow.log_metric("val_dssim", avg_dssim, step=epoch)
        if epoch > 0 and epoch % 50 == 0:
            print("Saving model checkpoints")
            torch.save(G.state_dict(), Path(f"models/coco/G_{epoch}.pt"))
            torch.save(D.state_dict(), Path(f"models/coco/D_{epoch}.pt"))
        visualize_batch(G, inputs, targets, epoch)
        generate_image(G, inputs[0], targets[0], epoch, f"epoch: {epoch}")
    print("Training finished!")
    return {
        "G_train_losses": G_train_losses,
        "D_train_losses": D_train_losses,
        "G_val_losses": G_val_losses,
        "D_val_losses": D_val_losses,
        "L1_train_losses": L1_train_losses,
        "L1_val_losses": L1_val_losses,
    }


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
    # Set our tracking server uri for logging
    # RUN mlflow server --host 127.0.0.1 --port 5053
    mlflow.set_tracking_uri(uri="http://127.0.0.1:5053")
    # print("Starting mlflow")
    # Create a new MLflow Experiment

    run_name = "condGAN_RGB_300_epoch_coco"
    mlflow.set_experiment("Colorization_coco")

    try:
        mlflow.start_run(run_name=run_name)
    except:
        mlflow.end_run()
        mlflow.start_run()

    img_paths_df = get_paths_df(dataset_path, ".jpg", 10240)  # limit to 10k images
    print(f"Found {len(img_paths_df)} images in the dataset")

    train_dataset, val_dataset, test_dataset = split_dataset(img_paths_df, 0.10, 0.10)

    train_dataloader = WrappedDataLoader(
        DataLoader(
            train_dataset, shuffle=True, batch_size=config.batch_size, num_workers=4
        ),
        device,
    )
    val_dataloader = WrappedDataLoader(
        DataLoader(
            val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=2
        ),
        device,
    )

    torch.set_float32_matmul_precision("high")

    D = Discriminator(3).to(device)
    G = Generator().to(device)

    optimizer_G = Adam(G.parameters(), lr=config.G_lr, betas=config.momentum_betas)
    optimizer_D = Adam(D.parameters(), lr=config.D_lr, betas=config.momentum_betas)

    # train the network
    losses_dict = fit(
        G,
        D,
        train_dataloader,
        val_dataloader,
        optimizer_G,
        optimizer_D,
        device,
    )

    plot_learning_curves(losses_dict)
    draw_network_architecture(GAN(G, D), torch.rand((1, 3, 512, 384)).to(device))
    # torch.save(G.state_dict(), Path(f"models/G_{run_name}.pt"))
    torch.save(G.state_dict(), Path("model.pt"))
    # torch.save(D.state_dict(), Path(f"models/D_{run_name}.pt"))
    mlflow.end_run()


def main() -> None:
    parser = ArgumentParser(description="Training script.")
    parser.add_argument("dataset_path", type=Path, help="Path to the dataset")
    args = parser.parse_args()
    training(args.dataset_path)


if __name__ == "__main__":
    main()
