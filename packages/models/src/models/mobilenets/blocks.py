import torch
import torch.nn as nn


class DepthwiseSeparableBlock(nn.Module):
    """
    Depthwise Separable Block
    This class implements a depthwise separable block as used in MobileNet architectures.
    It consists of a depthwise convolution followed by a pointwise convolution,
    both with batch normalization and ReLU activation.
    Architecture:
    - Depthwise Convolution -> Applies kernel (K, K, 1) for each input channel separately
    - Batch Normalization 1
    - ReLU Activation

    - Pointwise Convolution -> Applies kernel (1, 1, C') to combine the outputs of depthwise convolution
    with C' is the number of output channels.
    - Batch Normalization 2
    - ReLU Activation

    Notice: use in MobileNet for reducing the number of parameters and computations
    while maintaining performance.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super(DepthwiseSeparableBlock, self).__init__()
        self.depthwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_channels,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.pointwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Depthwise separable sonvolution
        out = self.depthwise(x)
        out = self.bn1(out)
        out = self.relu(out)

        # Pointwise convolution
        out = self.pointwise(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out


class InvertedResidualBlock(nn.Module):
    """
    Inverted Residual Block
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: int,
            expand_ratio: int,
    ):
        super(InvertedResidualBlock, self).__init__()
        self.stride = stride
        hidden_dim = int(in_channels * expand_ratio)  # expand mid channel in block
        self.use_residual_connection = self.stride == 1 and in_channels == out_channels

        layers = []
        # 1. Expand
        if expand_ratio != 1:
            layers.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=hidden_dim,
                    kernel_size=1,
                    bias=False,
                )
            )
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.ReLU6(inplace=True))

        # 2. Depthwise convolution
        layers.append(
            nn.Conv2d(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=hidden_dim,  # depthwise convolution
                bias=False,
            )
        )
        layers.append(nn.BatchNorm2d(hidden_dim))
        layers.append(nn.ReLU6(inplace=True))

        # 3. Projection
        layers.append(
            nn.Conv2d(
                in_channels=hidden_dim,
                out_channels=out_channels,
                kernel_size=1,
                bias=False,
            )
        )
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU6(inplace=True))

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_residual_connection:
            return x + self.block(x)
        else:
            return self.block(x)
