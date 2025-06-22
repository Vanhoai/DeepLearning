import torch
import torch.nn as nn


class SoftmaxClassifier(nn.Module):
    def __init__(self, input_dim=4096, num_classes=21):
        super(SoftmaxClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(input_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.classifier(x)
        return out
