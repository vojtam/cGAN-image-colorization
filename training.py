# STUDENT's UCO: 505941

# Description:
# This file should be used for performing training of a network
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
from torch.optim import Adam, Optimizer, RMSprop
from torch.utils.data import DataLoader
from torchview import draw_graph
from tqdm import tqdm

from dataset import (
    WrappedDataLoader,
    get_train_image_paths_df,
    lab_to_rgb_batch_np,
    lab_to_rgb_np,
    split_dataset,
)
from network import (
    Discriminator,
    Generator,
    compute_gradient_penalty,
    discriminator_loss,
    generator_loss,
    generator_loss_wgan,
)


def set_global_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


set_global_random_seed(42)


BCELogitsLoss = nn.BCEWithLogitsLoss()
L1Loss = torch.nn.L1Loss()


@dataclass
class config:
    G_lr: float = 0.00005
    D_lr: float = 0.00005
    batch_size: int = 32
    LAMBDA: int = 10
    momentum_betas: tuple[float, float] = (0.5, 0.999)
    epoch_num: int = 600


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


class TrainResults:
    def __init__(self):
        self.G_train_losses: list[float] = []
        self.L1_train_losses: list[float] = []
        self.critic_real_train_losses: list[float] = []
        self.critic_fake_train_losses: list[float] = []
        self.grad_penalties: list[float] = []

        self.train_G_loss = 0.0
        self.train_critic_real_loss = 0.0
        self.train_critic_fake_loss = 0.0
        self.train_L1_loss = 0.0
        self.train_gp = 0.0

    def update_train_step(self, G_loss, crit_real_loss, crit_fake_loss, L1, grad_pen):
        self.train_G_loss += G_loss
        self.train_critic_real_loss += crit_real_loss
        self.train_critic_fake_loss += crit_fake_loss
        self.train_L1_loss += L1
        self.train_gp += grad_pen

    def _reset(self):
        self.train_G_loss = 0.0
        self.train_critic_real_loss = 0.0
        self.train_critic_fake_loss = 0.0
        self.train_L1_loss = 0.0
        self.train_gp = 0.0

    def finish_train_epoch(
        self, epoch: int, n_batches, epoch_end_time: float, epoch_start_time: float
    ):
        self.G_train_losses.append(self.train_G_loss / n_batches)
        self.L1_train_losses.append(self.train_L1_loss / n_batches)
        self.critic_real_train_losses.append(self.train_critic_real_loss / n_batches)
        self.critic_fake_train_losses.append(self.train_critic_fake_loss / n_batches)
        self.grad_penalties.append(self.train_gp / n_batches)

        mlflow.log_metric("train_G_loss", self.G_train_losses[-1], step=epoch)
        mlflow.log_metric(
            "train_critic_real_loss", self.critic_real_train_losses[-1], step=epoch
        )
        mlflow.log_metric(
            "train_critic_fake_loss", self.critic_fake_train_losses[-1], step=epoch
        )
        mlflow.log_metric("train_L1_loss", self.L1_train_losses[-1], step=epoch)
        mlflow.log_metric("train_gp", self.grad_penalties[-1], step=epoch)
        mlflow.log_metric(
            "epoch_duration", epoch_end_time - epoch_start_time, step=epoch
        )
        print(
            f"Epoch: {epoch}, time_elapsed: {epoch_end_time - epoch_start_time:.2f} seconds | ",
            f"train_G_loss: {self.G_train_losses[-1]:.3f} | train_L1_loss: {self.L1_train_losses[-1]:.3f} | critic_real_loss: {self.critic_real_train_losses[-1]:.3f} |  critic_fake_loss: {self.critic_fake_train_losses[-1]:.3f} | train_GP: {self.grad_penalties[-1]:.3f}",
            sep="\n",
        )
        self._reset()


@torch.no_grad
def generate_image(
    model: Generator,
    input: Tensor,
    target: Tensor,
    epoch: int,
    is_lab: bool = True,
    plot_subtitle: str = "",
    save: bool = True,
    save_path=Path("gen_images/single"),
):
    # input has a shape C x H x W
    # target has a shape C x H x W
    model.eval()
    predicted = model(input.unsqueeze(0))  # add a batch dimension
    model.train()
    if is_lab:
        predicted_np = lab_to_rgb_np(input, predicted.squeeze())
        target_np = lab_to_rgb_np(input, target)
    else:
        target_np = target.permute(1, 2, 0).cpu().detach().numpy()
        predicted_np = predicted.squeeze().permute(1, 2, 0).cpu().detach().numpy()

    plt.figure(figsize=(10, 5))

    input_np = input.permute(1, 2, 0).cpu().detach().numpy()
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
    inputs: Tensor,
    targets: Tensor,
    epoch: int,
    save=True,
    save_path=Path("gen_images/batch"),
):
    model.eval()
    generator_output_ab = model(inputs)
    model.train()

    generated_imgs = lab_to_rgb_batch_np(inputs, generator_output_ab)
    ground_truth_imgs = lab_to_rgb_batch_np(inputs, targets)
    plt.figure(figsize=(10, 5))
    for i in range(5):
        ax = plt.subplot(3, 5, i + 1)
        ax.imshow(inputs[i][0].cpu(), cmap="gray")
        ax.axis("off")
        if i == 2:
            ax.set_title("Grayscale", fontsize=12)

        ax = plt.subplot(3, 5, i + 1 + 5)
        ax.imshow(ground_truth_imgs[i])
        ax.axis("off")
        if i == 2:
            ax.set_title("Ground Truth", fontsize=12)

        ax = plt.subplot(3, 5, i + 1 + 10)
        ax.imshow(generated_imgs[i])
        ax.axis("off")
        if i == 2:
            ax.set_title("Generated", fontsize=12)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    if save:
        save_path.mkdir(exist_ok=True)
        path = save_path / f"generated_images_{epoch}.png"
        print(f"Sample colorized image saved to {path}")
        plt.savefig(path)
        plt.close()
    else:
        plt.show()


def D_train_step(
    G: Generator,
    D: Discriminator,
    optimizer_D: Optimizer,
    inputs: Tensor,
    targets: Tensor,
    device: torch.device,
):
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        generator_output = G(inputs)  # 2 channels ab

        generated_conditional = torch.concat(
            (inputs, generator_output.detach()), dim=1
        )  # L + ab # Note to self: need to use detach here to keep the computational graphs separate -> otherwise backprop would break
        real_conditional = torch.concat((inputs, targets), dim=1)

        critic_generated_logits = D(generated_conditional)
        critic_real_logits = D(real_conditional)

    grad_penalty = compute_gradient_penalty(
        D, real_conditional, generated_conditional, device, 10
    )

    critic_real_loss = -torch.mean(critic_real_logits)
    critic_fake_loss = torch.mean(critic_generated_logits)
    D_loss = critic_fake_loss + critic_real_loss + grad_penalty

    optimizer_D.zero_grad()
    D_loss.backward()
    optimizer_D.step()

    return (
        generator_output,
        critic_real_loss.cpu(),
        critic_fake_loss.cpu(),
        grad_penalty.cpu(),
    )


def G_train_step(
    D: Discriminator,
    generator_output: Tensor,
    optimizer_G: Optimizer,
    inputs: Tensor,
    targets: Tensor,
    device: torch.device,
):
    generated_conditional = torch.concat((inputs, generator_output), dim=1)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        critic_generated_logits = D(generated_conditional)

    G_loss, G_L1_loss = generator_loss_wgan(
        critic_generated_logits, generator_output, targets, config.LAMBDA
    )
    G_total_loss = G_loss + G_L1_loss

    optimizer_G.zero_grad()
    G_total_loss.backward()
    optimizer_G.step()

    return G_loss.cpu(), G_L1_loss.cpu()


def train_step(
    G: Generator,
    D: Discriminator,
    optimizer_G: Optimizer | None,
    optimizer_D: Optimizer | None,
    inputs: Tensor,
    targets: Tensor,
    trainResults: TrainResults,
    device: torch.device,
):
    # 1) obtain the "fake" color ab image by running L inputs through generator
    # 2) Discriminator's turn:
    #      - run Discriminator on generated full image (L + generated ab - detach)
    #      - run Discriminator on real full image (concat inputs with targets)
    #      - compute Discriminator loss
    #      - backward + step with discriminator
    # 3) Generator's turn:
    #      - run the Discriminator on generated full iamge but don't detach (so that the generator can learn from D's judgement)
    #      - compute Generator loss

    #      - backward + step with generator
    gen_output, critic_real_loss, critic_fake_loss, grad_penalty = D_train_step(
        G, D, optimizer_D, inputs, targets, device
    )
    G_loss, G_L1_loss = G_train_step(
        D, gen_output, optimizer_G, inputs, targets, device
    )

    trainResults.update_train_step(
        G_loss, critic_real_loss, critic_fake_loss, G_L1_loss, grad_penalty
    )


@torch.no_grad
def valid_step(
    G: Generator,
    D: Discriminator,
    inputs: Tensor,
    targets: Tensor,
    device: torch.device,
    epoch: int,
):
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        generator_output = G(inputs)  # 2 channels ab

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

    # if epoch % 20 == 0:
    #     pred_RGB = lab_to_rgb_batch_np(inputs, generator_output) * 255
    #     ref_RGB = lab_to_rgb_batch_np(inputs, targets) * 255
    #     dssim_score = 0
    #     for ref_img, pred_img in zip(ref_RGB, pred_RGB):
    #         dssim_score += dssim(ref_img.astype(np.uint8), pred_img.astype(np.uint8))
    #     dssim_score /= inputs.shape[0]
    #     return FitStepResult(G_total_loss, G_loss, G_L1_loss, D_loss, dssim=dssim_score)

    # return FitStepResult(G_total_loss, G_loss, G_L1_loss, D_loss)


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
    trainResults = TrainResults()

    for epoch in range(config.epoch_num):
        G.train()
        D.train()
        torch.cuda.synchronize()
        epoch_start_time = time.time()

        n_batches = 0
        for inputs, targets in tqdm(train_dataloader):
            # torch.compiler.cudagraph_mark_step_begin()
            n_batches += 1
            train_step(
                G,
                D,
                optimizer_G,
                optimizer_D,
                inputs,
                targets,
                trainResults,
                device,
            )

        # VALIDATION
        G.eval()
        D.eval()

        for inputs, targets in tqdm(val_dataloader):
            # torch.compiler.cudagraph_mark_step_begin()
            # step_result = valid_step(
            #     G,
            #     D,
            #     inputs,
            #     targets,
            #     device,
            #     epoch,
            # )

            pass

        torch.cuda.synchronize()
        epoch_end_time = time.time()

        trainResults.finish_train_epoch(
            epoch, n_batches, epoch_end_time, epoch_start_time
        )
        visualize_batch(G, inputs, targets, epoch)
        generate_image(G, inputs[0], targets[0], epoch, f"epoch: {epoch}")

        # if epoch % 20 == 0:
        #     avg_dssim = dssim / n_val_batches
        #     print(f"average dssim score : {avg_dssim}")
        #     mlflow.log_metric("val_dssim", avg_dssim, step=epoch)

    print("Training finished!")
    return trainResults


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
    # RUN mlflow server --host 127.0.0.1 --port 5051
    mlflow.set_tracking_uri(uri="http://127.0.0.1:5051")
    print("Starting mlflow")
    # Create a new MLflow Experiment

    run_name = "condGAN_17_600_epoch"
    mlflow.set_experiment("Colorization_01")

    try:
        mlflow.start_run(run_name=run_name)
    except:
        mlflow.end_run()
        mlflow.start_run()

    img_paths_df = get_train_image_paths_df(dataset_path)
    # img_paths_df = get_image_paths_df(dataset_path, n=5000)

    train_dataset, val_dataset, _ = split_dataset(img_paths_df, 0.10, 0.1)

    train_dataloader = WrappedDataLoader(
        DataLoader(
            train_dataset, shuffle=True, batch_size=config.batch_size, num_workers=10
        ),
        device,
    )
    val_dataloader = WrappedDataLoader(
        DataLoader(
            val_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4
        ),
        device,
    )

    torch.set_float32_matmul_precision("high")

    D = Discriminator(3).to(device)
    G = Generator().to(device)

    optimizer_G = Adam(G.parameters(), lr=config.G_lr, betas=config.momentum_betas)
    optimizer_D = RMSprop(D.parameters(), lr=config.D_lr)

    # train the network
    train_losses, val_losses = fit(
        G,
        D,
        train_dataloader,
        val_dataloader,
        optimizer_G,
        optimizer_D,
        device,
    )

    torch.save(G.state_dict(), Path(f"models/G_{run_name}.pt"))
    torch.save(D.state_dict(), Path(f"models/D_{run_name}.pt"))
    mlflow.end_run()


# #### code below should not be changed ############################################################################


def main() -> None:
    parser = ArgumentParser(description="Training script.")
    parser.add_argument("dataset_path", type=Path, help="Path to the dataset")
    args = parser.parse_args()
    training(args.dataset_path)


if __name__ == "__main__":
    main()
