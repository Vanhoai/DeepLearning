import torch
from torchvision.transforms import functional as F


class ToTensor(object):
    def __call__(self, sample):
        region, label = sample

        # Convert region to tensor
        # Convert HWC to CHW
        region = torch.from_numpy(region).permute(2, 0, 1).float()
        label = torch.from_numpy(label).long()

        return region, label


class Normalize(object):
    def __call__(self, sample):
        region, label = sample

        # Normalize the region tensor
        region = F.normalize(
            region,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        return region, label
