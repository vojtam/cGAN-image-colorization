# STUDENT's UCO: 000000

# Description:
# This file should contain network class. The class should subclass the torch.nn.Module class.

import torch
from torch import Tensor, nn

bce = nn.BCELoss()
l1 = nn.L1Loss()


@torch.no_grad
def generator_loss(
    discriminator_G_output: Tensor,
    generated_output: Tensor,
    target: Tensor,
    LAMBDA: int = 100,
):
    G_loss = bce(torch.ones_like(discriminator_G_output), discriminator_G_output)
    l1_loss = l1(generated_output, target)

    total_G_loss = G_loss + l1_loss * LAMBDA

    total_G_loss.requires_grad = True
    return total_G_loss, G_loss, l1_loss


@torch.no_grad
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
    total_D_loss.requires_grad = True
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
            if i < len(skip_connections):
                x = up_layer(x, skip_connections[i])
            else:
                print("NO SKIP CONNECT")
                x = up_layer(x)
        x = self.output_layer(x)
        return x


class Discriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # Ck = Convolution-BatchNorm-ReLU
        # discriminator: C64-C128-C256-C512 -> classification head (sigmoid)

        self.down1 = DownBlock(6, 64, False)
        self.down2 = DownBlock(64, 128)
        self.down3 = DownBlock(128, 256)
        self.down4 = DownBlock(256, 512)

        self.last = nn.Conv2d(512, 1, kernel_size=(4, 4), stride=1)

    def forward(self, x_input: Tensor, x_target: Tensor) -> Tensor:
        x = torch.concat((x_input, x_target), dim=1)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)

        x = nn.functional.sigmoid(self.last(x))
        return x


class DownBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, batch_norm: bool = True
    ) -> None:
        super().__init__()
        layers = [
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(4, 4),
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.LeakyReLU(0.2),
        ]

        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
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
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
        )

        if use_dropout:
            self.upsample.append(nn.Dropout2d(0.5))

    def forward(self, x: Tensor, residual_x: Tensor) -> Tensor:
        x = self.upsample(x)
        return torch.cat([x, residual_x], dim=1)
