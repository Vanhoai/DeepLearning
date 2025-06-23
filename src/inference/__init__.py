import torch
from models import BackboneWithFPN


def main() -> None:
    backbone = BackboneWithFPN(backbone_name="resnet50", pretrained=True)

    # fake input tensors shape (batch_size, channels, height, width)
    x = torch.randn(1, 3, 512, 512)
    output = backbone(x)

    # print all p in output
    for i in range(len(output)):
        key = "p" + str(i + 2)
        print(f"Output {i} shape: {output[key].shape}")
