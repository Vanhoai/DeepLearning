import cv2
from numpy.typing import NDArray
from typing import Tuple, List
import torch
import torch.nn as nn
from torchvision.models import alexnet, AlexNet_Weights
from torch.utils.data import Dataset


class SelectiveSearch:
    """
    Selective Search for Region Proposals
    This class implements the Selective Search algorithm to generate region proposals
    from an input image. It uses OpenCV's ximgproc module for segmentation.
    """

    def __init__(self):
        self.ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

    def proposals(self, image: NDArray, mode="fast") -> List[Tuple[int, int, int, int]]:
        """
        Generate region proposals using Selective Search.
        Args:
            image (NDArray): Input image in BGR format.
            mode (str): Mode for selective search, either "fast" or "quality".
        Returns:
            List[Tuple[int, int, int, int]]: List of bounding boxes in the format (x, y, width, height).
        """
        self.ss.setBaseImage(image)
        if mode == "fast":
            self.ss.switchToSelectiveSearchFast()
        elif mode == "quality":
            self.ss.switchToSelectiveSearchQuality()
        else:
            # Default Mode is Fast
            self.ss.switchToSelectiveSearchFast()

        rects = self.ss.process()

        # Filter proposals by size and aspect ratio
        filtered_rects = []
        h, w = image.shape[:2]

        for rect in rects:
            x, y, width, height = rect

            # Ignore rectangles that are too small or too large
            if width < 20 or height < 20 or width * 0.8 > w or height > h * 0.8:
                continue

            # Ignore rectangles with extreme aspect ratios
            aspect_ratio = width / height
            if aspect_ratio < 0.2 or aspect_ratio > 5.0:
                continue

            filtered_rects.append((x, y, width, height))

        return filtered_rects[:2000]  # Limit to 2000 proposals


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


class RCNNDataset(Dataset):
    """
    Custom Dataset for R-CNN
    This class is used to create a dataset for training or evaluating an RCNN model.
    It should be implemented with methods to load images and their corresponding labels.
    """

    def __init__(self, images, boxes, labels, transform=None):
        """
        Args:
            images (List[NDArray]): List of images.
            boxes (List[List[Tuple[int, int, int, int]]]): List of bounding boxes for each image.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.images = images
        self.boxes = boxes
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        box = self.boxes[idx]
        label = self.labels[idx]

        # Crop the region from the image
        x, y, w, h = box
        region = image[y:y + h, x:x + w]

        # Resize to 227x227 for AlexNet
        region = cv2.resize(region, (227, 227))
        region = cv2.cvtColor(region, cv2.COLOR_BGR2RGB)

        if self.transform:
            region = self.transform(region)

        return region, label
