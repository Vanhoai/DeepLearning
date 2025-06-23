import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from collections import OrderedDict
import torch.nn.functional as functional


class BackboneWithFPN(nn.Module):
    """
    ResNet backbone with Feature Pyramid Network (FPN)
    """

    def __init__(self, backbone_name="resnet50", pretrained=True):
        super(BackboneWithFPN, self).__init__()

        # Load pretrained ResNet backbone
        if backbone_name == "resnet50":
            backbone = resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)
        else:
            raise NotImplementedError(f"Backbone {backbone_name} not supported")

        # Extract feature layers
        self.layer0 = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool
        )

        # ????????
        self.layer1 = backbone.layer1  # 1/4
        self.layer2 = backbone.layer2  # 1/8
        self.layer3 = backbone.layer3  # 1/16
        self.layer4 = backbone.layer4  # 1/32

        # FPN layers
        in_channels = [256, 512, 1024, 2048]  # ResNet50 output each layer
        out_channel = 256  # FPN output channels

        # Lateral connections
        self.lateral_convs = nn.ModuleList()

        # Output connections
        self.output_convs = nn.ModuleList()

        for in_channel in in_channels:
            # change size from in_channel to out_channel
            self.lateral_convs.append(nn.Conv2d(in_channel, out_channel, kernel_size=1))

            # keep the same size
            self.output_convs.append(nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1))

        # After initialization above, we have:
        # lateral convs:
        # 0: [256 -> 256]
        # 1: [512 -> 256]
        # 2: [1024 -> 256]
        # 3: [2048 -> 256]
        # output convs:
        # [256 -> 256] x 4

        # Initialize weights
        for m in self.lateral_convs.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        for m in self.output_convs.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Bottom-up pathway
        c1 = self.layer0(x)
        c2 = self.layer1(c1)  # 1/4 resolution
        c3 = self.layer2(c2)  # 1/8 resolution
        c4 = self.layer3(c3)  # 1/16 resolution
        c5 = self.layer4(c4)  # 1/32 resolution

        # Top-down pathway and lateral connections
        features = OrderedDict()

        lateral_5 = self.lateral_convs[3](c5)
        p5 = self.output_convs[3](lateral_5)
        features["p5"] = p5  # 1/32

        # Top-down pathway with lateral connections
        lateral_4 = self.lateral_convs[2](c4)
        top_down_4 = functional.interpolate(lateral_5, size=lateral_4.shape[-2:], mode="nearest")
        p4 = self.output_convs[2](lateral_4 + top_down_4)
        features["p4"] = p4  # 1/16

        lateral_3 = self.lateral_convs[1](c3)
        top_down_3 = functional.interpolate(p4, size=lateral_3.shape[-2:], mode="nearest")
        p3 = self.output_convs[1](lateral_3 + top_down_3)
        features["p3"] = p3  # 1/8

        lateral_2 = self.lateral_convs[0](c2)
        top_down_2 = functional.interpolate(p3, size=lateral_2.shape[-2:], mode="nearest")
        p2 = self.output_convs[0](lateral_2 + top_down_2)
        features["p2"] = p2  # 1/4

        return features  # p2, p3, p4, p5 in descending order of resolution
