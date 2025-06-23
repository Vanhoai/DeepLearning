import torch
import torch.nn as nn
from .blocks import DepthwiseSeparableBlock, InvertedResidualBlock


class MobileNetV1(nn.Module):
    def __init__(self, num_classes: int = 1000):
        super(MobileNetV1, self).__init__()

        # Initial convolution layer
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
        )

        # Depthwise separable convolutions
        self.features = nn.Sequential(
            self.features,
            DepthwiseSeparableBlock(32, 64, 1),
            DepthwiseSeparableBlock(64, 128, 2),
            DepthwiseSeparableBlock(128, 128, 1),
            DepthwiseSeparableBlock(128, 256, 2),
            DepthwiseSeparableBlock(256, 256, 1),
            DepthwiseSeparableBlock(256, 512, 2),
            DepthwiseSeparableBlock(512, 512, 1),
            DepthwiseSeparableBlock(512, 512, 1),
            DepthwiseSeparableBlock(512, 512, 1),
            DepthwiseSeparableBlock(512, 512, 1),
            DepthwiseSeparableBlock(512, 512, 1),
            DepthwiseSeparableBlock(512, 1024, 2),
            DepthwiseSeparableBlock(1024, 1024, 1),
        )

        # Average pooling and classification layer
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Linear(1024, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000):
        super(MobileNetV2, self).__init__()

        # (t, c, n, s): (expand_ratio, out_channels, num_blocks, stride)
        self.configs = [
            (1, 16, 1, 1),
            (6, 24, 2, 2),
            (6, 32, 3, 2),
            (6, 64, 4, 2),
            (6, 96, 3, 1),
            (6, 160, 3, 2),
            (6, 320, 1, 1),
        ]

        # initial layer
        input_channels = 32
        layers = [
            nn.Conv2d(
                3,  # BGR
                input_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(input_channels),
            nn.ReLU6(inplace=True),
        ]

        # Inverted Residual blocks
        for t, c, n, s in self.configs:
            output_channel = c
            for i in range(n):
                stride = s if i == 0 else 1
                layers.append(InvertedResidualBlock(input_channels, output_channel, stride, t))  # type: ignore
                input_channels = output_channel

        # Final Conv 1x1
        last_channel = 1280
        layers.append(
            nn.Conv2d(input_channels, last_channel, kernel_size=1, bias=False)
        )
        layers.append(nn.BatchNorm2d(last_channel))
        layers.append(nn.ReLU6(inplace=True))
        self.features = nn.Sequential(*layers)

        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(last_channel, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
