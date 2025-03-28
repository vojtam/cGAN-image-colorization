# STUDENT's UCO: 505941

# Description:
# This file should contain network class. The class should subclass the torch.nn.Module class.

from pathlib import Path

import torch
from torch import Tensor, nn

bce = nn.BCELoss()
l1 = nn.L1Loss()


def generator_loss(
    discriminator_G_output: Tensor,
    generated_output: Tensor,
    target: Tensor,
    LAMBDA: int = 100,
):
    G_loss = bce(torch.ones_like(discriminator_G_output), discriminator_G_output)
    l1_loss = l1(generated_output, target)

    total_G_loss = G_loss + l1_loss * LAMBDA
    return total_G_loss, G_loss, l1_loss


def discriminator_loss(
    discriminator_real_output: Tensor, discriminator_G_output: Tensor
):
    real_loss = bce(
        torch.ones_like(discriminator_real_output), discriminator_real_output
    )
    generated_loss = bce(
        torch.zeros_like(discriminator_G_output), discriminator_G_output
    )

    total_D_loss = real_loss + generated_loss
    return total_D_loss


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
                out_channels=3,
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
        self.layers = nn.Sequential(
            DownBlock(input_channels, 64, False),
            DownBlock(64, 128),
            DownBlock(128, 256),
            DownBlock(256, 512, stride=1),
            nn.Conv2d(512, 1, kernel_size=(4, 4), stride=1, padding=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = nn.functional.sigmoid(self.layers(x))
        return x


class GAN(nn.Module):
    def __init__(
        self, G: Generator, D: Discriminator, run_name: str = "conditional_GAN_01"
    ) -> None:
        super().__init__()
        self.G = G
        self.D = D
        self.run_name = run_name

    def save_model(self, dir_to_save: Path):
        G_filename = Path(f"G_{self.run_name}_model.pt")
        D_filename = Path(f"D_{self.run_name}_model.pt")

        try:
            torch.save(self.G.state_dict(), dir_to_save / G_filename)
            torch.save(self.D.state_dict(), dir_to_save / D_filename)
            print(f"Saved the Generator model to {dir_to_save / G_filename}")
            print(f"Saved the Discriminator model to {dir_to_save / D_filename}")
        except Exception as e:
            print(f"An ERROR occurred while saving: {e}")

    def compute_loss(
        self,
        generated_image: Tensor,
        D_real_output: Tensor,
        D_generated_output: Tensor,
        targets: Tensor,
    ):
        total_G_loss, G_loss, G_l1_loss = self.G.generator_loss(
            D_generated_output, generated_image, targets
        )

        D_loss = self.D.discriminator_loss(D_real_output, D_generated_output)

        # return FitStepResult(total_G_loss, G_loss, G_l1_loss, D_loss)

    def forward(self, inputs: Tensor, targets: Tensor):
        generator_output = self.G(inputs)
        inputs = inputs.repeat(1, 3, 1, 1)
        discriminator_real_output = self.D(inputs, targets)
        discriminator_generated_output = self.D(inputs, generator_output)

        return self.compute_loss(
            generator_output,
            discriminator_real_output,
            discriminator_generated_output,
            targets,
        )


class DownBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        batch_norm: bool = True,
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
