# STUDENT's UCO: 000000

# Description:
# This file should contain network class. The class should subclass the torch.nn.Module class.

import torch
from torch import Tensor, nn


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
                in_channels=down_filters[-1],
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

        for i in range(len(self.up_layers)):
            x = self.up_layers[i](x, skip_connections[len(self.up_layers) - i - 1])

        return x


class Discriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor: ...


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
