# STUDENT's UCO: 000000

# Description:
# This file should contain network class. The class should subclass the torch.nn.Module class.

from torch import Tensor, nn


class Generator(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    # Each block in the encoder is: Convolution -> Batch normalization -> Leaky ReLU

    def forward(self, x: Tensor) -> Tensor: ...


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


class DownSampler(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        filters = [64, 64, 128, 256, 512, 512]

        layers = [DownBlock(1, filters[0], False)] + [
            DownBlock(filters[i - 1], filters[i], True)
            for i in range(1, len(filters), 1)
        ]

        self.down = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.down(x)


class UpSampler(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor: ...
