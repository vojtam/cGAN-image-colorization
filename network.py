# STUDENT's UCO: 505941

# Description:
# This file should contain network class. The class should subclass the torch.nn.Module class.


import torch
from torch import Tensor, nn

BCELogitsLoss = nn.BCEWithLogitsLoss(reduction="mean")
L1Loss = torch.nn.L1Loss(reduction="mean")

from torch.autograd import grad


# attribution: https://discuss.pytorch.org/t/wgan-gradient-penalty-error-even-with-retain-graph-true/113266
def compute_gradient_penalty(
    critic: nn.Module,
    real_samples: Tensor,
    fake_samples: Tensor,
    device: torch.device,
    lambda_gp: float = 10.0,
) -> Tensor:
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    # Shape: (batch_size, 1, 1, 1) to broadcast correctly
    alpha = torch.rand(
        (real_samples.size(0), 1, 1, 1), device=device, requires_grad=False
    )

    # Get random interpolation between real and fake samples
    # Shape: (batch_size, channels, height, width)
    interpolated_images = (
        alpha * real_samples + ((1 - alpha) * fake_samples)
    ).requires_grad_(True)

    critic_score_interpolated = critic(interpolated_images)  # logits

    # Use fake gradients (batch of ones) for backprop, matching the critic output shape
    fake_grad_outputs = torch.ones_like(
        critic_score_interpolated, requires_grad=False, device=device
    )

    gradients = grad(
        outputs=critic_score_interpolated,
        inputs=interpolated_images,
        grad_outputs=fake_grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]  # grad returns a tuple, we only need the gradient w.r.t. interpolates

    gradients = gradients.view(gradients.size(0), -1)

    # Calculate the norm and the penalty
    # Add small epsilon for stability if needed, though norm usually handles it
    gradient_norm = gradients.norm(2, dim=1)  # L2 norm for each sample in batch
    gradient_penalty = (
        (gradient_norm - 1) ** 2
    ).mean()  # Mean squared difference from 1

    return lambda_gp * gradient_penalty  # Scale by lambda_gp


def critic_loss_wgan(critic_real_output: Tensor, critic_fake_output: Tensor) -> Tensor:
    """
    Calculates the base WGAN critic loss.
    Note: Gradient penalty is added separately.
    """
    # We want to maximize D(real) - D(fake) which is equivalent to minimizing D(fake) - D(real)
    loss_c = torch.mean(critic_fake_output) - torch.mean(critic_real_output)
    return loss_c


def generator_loss_wgan(
    critic_fake_output: Tensor,  # logits of the discriminator on the cat(inputs_L, generated_ab)
    generator_output: Tensor,
    targets: Tensor,
    lambda_l1: int = 100,
) -> tuple[Tensor, Tensor]:
    """
    Calculates the WGAN generator loss components.
    - Adversarial loss: Maximize D(fake) -> Minimize -D(fake)
    Returns the total loss for the generator and the L1 component separately.
    """
    # Adversarial Loss: We want critic to output high values for fake images
    adversarial_loss_g = -torch.mean(critic_fake_output)

    # L1 Loss (same as before)
    l1_loss_g = L1Loss(generator_output, targets) * lambda_l1

    return adversarial_loss_g, l1_loss_g


def generator_loss(
    discriminator_generated_output: Tensor,
    generator_output: Tensor,
    targets: Tensor,
    LAMBDA: int = 100,
):
    labels = torch.ones_like(discriminator_generated_output, requires_grad=False)
    bce_G_loss = BCELogitsLoss(discriminator_generated_output, labels)

    L1_G_loss = L1Loss(generator_output, targets) * LAMBDA
    return bce_G_loss, L1_G_loss


def discriminator_loss(
    discriminator_generated_output: Tensor,
    discriminator_real_output: Tensor,
    smoothing_factor: float | None = 0.9,
):
    fake_labels = torch.zeros_like(discriminator_generated_output, requires_grad=False)
    real_labels = torch.ones_like(discriminator_real_output, requires_grad=False)
    if smoothing_factor is not None:
        real_labels *= smoothing_factor

    D_fake_loss = BCELogitsLoss(
        discriminator_generated_output,
        fake_labels,
    )
    D_real_loss = BCELogitsLoss(discriminator_real_output, real_labels)

    D_loss = D_fake_loss + D_real_loss
    return D_loss


class Generator(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        # Each block in the encoder is: Convolution -> Batch normalization -> Leaky ReLU

        down_filters = [1, 64, 64, 128, 256, 256, 512]
        use_batch_norm = [False, False, True, True, True, True, False]

        self.down_layers = nn.ModuleList(
            [
                DownBlock(down_filters[i], down_filters[i + 1], use_batch_norm[i + 1])
                for i in range(len(down_filters) - 1)
            ]
        )

        bottleneck_out_channels = 512
        self.bottleneck = DownBlock(down_filters[-1], bottleneck_out_channels, False)

        self.up_layers = nn.ModuleList()
        up_filters = [512, 256, 256, 128, 64, 64]

        in_channels = bottleneck_out_channels
        for i in range(len(up_filters)):
            out_channels = up_filters[i]
            if i > 0:
                in_channels *= 2
            self.up_layers.append(UpBlock(in_channels, out_channels, i < 3))
            in_channels = up_filters[i]

        self.output_layer = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=up_filters[-1] * 2,
                out_channels=2,
                kernel_size=(4, 4),
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.Tanh(),
        )

    def forward(self, x: Tensor) -> Tensor:
        skip_connections = []

        for down_layer in self.down_layers:
            x = down_layer(x)
            skip_connections.append(x)

        x = self.bottleneck(x)

        skip_connections = skip_connections[::-1]
        for i, up_layer in enumerate(self.up_layers):
            x = up_layer(x, skip_connections[i])
        x = self.output_layer(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, input_channels: int) -> None:
        super().__init__()
        # Ck = Convolution-BatchNorm-ReLU
        # discriminator: C64-C128-C256-C512 -> classification head (sigmoid)
        layers = nn.ModuleList(
            [
                DownBlock(input_channels, 64, False),
                DownBlock(64, 128, False, False),
                DownBlock(128, 256, False, False),
                DownBlock(256, 512, False, False, stride=1),
                nn.Conv2d(512, 1, kernel_size=(4, 4), stride=1, padding=1),
            ]
        )

        for layer in layers[:-1]:
            nn.utils.parametrizations.spectral_norm(layer.downsample[0])
        nn.utils.parametrizations.spectral_norm(layers[-1])
        self.discriminator_layers = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.discriminator_layers(x)
        return x


class DownBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        batch_norm: bool = True,
        instance_norm: bool = False,
        stride: int = 2,
    ) -> None:
        super().__init__()
        layers = [
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(4, 4),
                stride=stride,
                padding=1,
                bias=False,
            ),
        ]

        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        elif instance_norm:
            layers.append(nn.InstanceNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2))
        self.downsample = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.downsample(x)


class UpBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, use_dropout: bool = True
    ) -> None:
        super().__init__()

        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(4, 4),
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

        if use_dropout:
            self.upsample.append(nn.Dropout2d(0.5))

    def forward(self, x: Tensor, residual_x: Tensor) -> Tensor:
        x = self.upsample(x)
        # Note to self:
        # in case the input image had odd resolution in any dimensions,
        # due to using mostly stride 2, the image resolution might have gotten rounded at some point
        # and the dimensions of the skip with the x might not be compatible =>
        # => interpolating the single value should solve it
        # if the shapes match, the tensor will be as it was
        residual_x = torch.nn.functional.interpolate(
            residual_x, size=x.shape[2:], mode="nearest"
        )
        return torch.cat([x, residual_x], dim=1)
