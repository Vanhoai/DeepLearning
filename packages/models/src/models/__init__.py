from .blocks import BottleneckBlock, ResidualBlock
from .res_nets import ResNet
from .mobile_nets import MobileNetV1, MobileNetV2
from torchinfo import summary

# ResNet-18: 11,681,832
# ResNet-50: 25,549,352
# MobileNetV1: 4,231,976
# MobileNetV2: 3,504,872


def main() -> None:
    model = MobileNetV2()
    summary(model, (1, 3, 224, 224))
