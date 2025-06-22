import torch
import torch.nn as nn
from torchvision.models import alexnet, AlexNet_Weights


class FeatureExtractor(nn.Module):
    """
    CNN Feature Extractor based on AlexNet
    This class is used to extract features from images using a pre-trained AlexNet model.
    Notice:
        - The model is set to evaluation mode.
        - The input images should be normalized to the same scale as the pre-trained model.
        - The output features are flattened and returned as a tensor.
    """

    def __init__(self, pretrained=True):
        super(FeatureExtractor, self).__init__()

        # Load pretrained AlexNet
        alexnet_model = alexnet(weights=AlexNet_Weights.DEFAULT)

        # Remove the last fully connected layer
        self.features = alexnet_model.features
        self.avgpool = alexnet_model.avgpool
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True)
        )

        # Copy weights from pretrained model
        if pretrained:
            for i, layer in enumerate(self.features):
                if isinstance(layer, nn.Linear):
                    layer.weight.data = alexnet_model.classifier[i].weight.data
                    layer.bias.data = alexnet_model.classifier[i].bias.data

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.features(x)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out
