import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    # This is used in ResNet-50, ResNet-101, and ResNet-152 architectures.
    expansion = 1

    """
    Residual Block
    This class implements a residual block as described in the ResNet architecture.
    It consists of two convolutional layers with batch normalization and ReLU activation.
    Architecture:
    - Convolutional Layer 1 -> Reduces spatial dimensions if stride > 1
    - Batch Normalization 1
    - ReLU Activation
    - Convolutional Layer 2 -> Maintains spatial dimensions
    - Batch Normalization 2
    - Shortcut Connection -> Bypasses the convolutional layers if input and output dimensions differ
    - ReLU Activation

    64 -> 128 -> 256 -> 512 channels

    Stage 1:
    ResidualBlock(64, 64, stride=1) x 2 -> 64 channels (expansion = 1) so the output same as input

    Stage 2:
    ResidualBlock(64, 128, stride=2)
    ResidualBlock(128, 128, stride=1)
    ....

    Looking layers in ResidualBlock:
    - conv1: First convolutional layer that is use in and out channels and stride from params
    so this layer is used to reduce the spatial dimensions of the input.
    - bn1: Batch normalization layer after the first convolution.
    - conv2: Second convolutional layer that keeps the same number of channels as the first
    and has a kernel size of 3x3 with padding to maintain the spatial dimensions.
    - bn2: Batch normalization layer after the second convolution.
    - shortcut: A shortcut connection that allows the input to bypass the convolutional layers.
    If the input and output channels do not match, a 1x1 convolution is applied
    to adjust the dimensions.

    Notice: this class only uses in ResNet-18 and ResNet-34 architectures.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=self.expansion * out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(identity)
        out = self.relu(out)
        return out


class BottleneckBlock(nn.Module):
    expansion = 4
    """
    Bottleneck Block
    This class implements a bottleneck block as described in the ResNet architecture.
    It consists of three convolutional layers with batch normalization and ReLU activation.
    Architecture:
    - Convolutional Layer 1 (1 x 1) -> Reduces the number of channels
    - Batch Normalization 1
    - ReLU Activation
    - Convolutional Layer 2 (3 x 3) -> Maintains spatial dimensions
    - Batch Normalization 2
    - ReLU Activation
    - Convolutional Layer 3 (1 x 1) -> Increases the number of channels
    - Batch Normalization 3
    - Shortcut Connection -> Bypasses the convolutional layers if input and output dimensions differ
    - ReLU Activation

    64 -> 128 -> 256 -> 512 channels
    3, 4, 6, 3

    strides: [1/2 in first and 1 for all others blocks]

    Stage 1: 3 blocks
    BottleneckBlock(64, 64, stride=1)
    - Conv1: 1 x 1 -> 64 channels
    - Conv2: 3 x 3 -> 64 channels
    - Conv3: 1 x 1 -> 256 channels (because multiplied by expansion factor 4)

    BottleneckBlock(256, 64, stride=1) 256 = 64 * 4
    BottleneckBlock(256, 64, stride=1) -> 256 = 64 * 4

    Stage 2: 4 blocks
    BottleneckBlock(256, 128, stride=2) 256 != 128 * 4 = 512
    BottleneckBlock(512, 128, stride=1) x 3 -> 512 = 128 * 4

    Stage 3: 6 blocks
    BottleneckBlock(512, 256, stride=2) 512 != 256 * 4 = 1024
    BottleneckBlock(1024, 256, stride=1) x 5 -> 1024 = 256 * 4

    ....
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super(BottleneckBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=self.expansion * out_channels,
            kernel_size=1,
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(self.expansion * out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=self.expansion * out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += self.shortcut(identity)
        out = self.relu(out)
        return out


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
