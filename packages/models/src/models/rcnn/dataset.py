import cv2
from typing import Tuple, List
from torch.utils.data import Dataset
from numpy.typing import NDArray


class RegionsDataset(Dataset):
    def __init__(self, regions: List[NDArray], labels: List[int], transform=None):
        self.regions = regions
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.regions)

    def __getitem__(self, idx: int):
        region = self.regions[idx]

        # Assuming region is a cropped image, resize it to 227x227
        region = cv2.resize(region, (227, 227))
        region = cv2.cvtColor(region, cv2.COLOR_BGR2RGB)

        if self.transform:
            region = self.transform(region)

        label = self.labels[idx]
        return region, label


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
