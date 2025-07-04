import torch
from torch import Tensor, nn

BCELogitsLoss = nn.BCEWithLogitsLoss()
L1Loss = torch.nn.L1Loss()


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
    smoothing_factor: float | None = None,
):
    fake_labels = torch.zeros_like(discriminator_generated_output, requires_grad=False)
    real_labels = torch.ones_like(discriminator_real_output, requires_grad=False)
    if smoothing_factor is not None:
        real_labels *= smoothing_factor

    D_fake_loss = BCELogitsLoss(discriminator_generated_output, fake_labels)
    D_real_loss = BCELogitsLoss(discriminator_real_output, real_labels)

    D_loss = (
        D_fake_loss + D_real_loss
    ) / 2  # divide by 2 per paper to slow down discriminator learning
    return D_loss


class Generator(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        # Each block in the encoder is: Convolution -> Batch normalization -> Leaky ReLU

        down_filters = [
            3,
            64,
            64,
            128,
            256,
            256,
            512,
        ]
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
            # nn.Tanh(), # I don't really know why but replacing the tanh with sigmoid and not normalizing to [-1, 1] made the resulting images look better
            # probably bug in my code
            nn.Sigmoid(),
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
                DownBlock(64, 128, stride=1),
                DownBlock(128, 256),
                DownBlock(256, 512, stride=1),
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


# This torch module class is only for the final architecture visualization
# so that I can visualize both the Generator and Discriminator in the same image
class GAN(nn.Module):
    def __init__(self, G: Generator, D: Discriminator):
        super().__init__()
        self.G = G
        self.D = D

    def forward(self, x: Tensor):
        xg_out = self.G(x)
        xd_out = self.D(x.repeat([1, 2, 1, 1]))
        return xg_out, xd_out


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
        return torch.cat([x, residual_x], dim=1)
