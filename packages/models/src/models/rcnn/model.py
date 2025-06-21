import torch
import torch.nn as nn
from torchvision import transforms
from typing import List, Tuple
from numpy.typing import NDArray
from .functions import SelectiveSearch, FeatureExtractor, SoftmaxClassifier
import multiprocessing

type BoundingBox = Tuple[int, int, int, int]  # (x, y, width, height)


class RCNN:
    def __init__(self, num_classes=21):  # 20 + 1 for background
        self.num_classes = num_classes
        self.selective_search = SelectiveSearch()
        self.feature_extractor = FeatureExtractor(pretrained=True)
        self.classifier = SoftmaxClassifier(input_dim=4096, num_classes=num_classes)

        # Training parameters
        self.device = torch.accelerator.current_accelerator()
        self.feature_extractor.to(self.device)
        self.classifier.to(self.device)

        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((227, 227)),  # Resize to 227x227 for AlexNet
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

        # Set eval mode for feature extractor avoid gradient computation
        self.feature_extractor.eval()

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

    def extract_features(self, regions: List[NDArray]) -> torch.Tensor:
        if len(regions) == 0:
            return torch.empty(0, 4096)

        batch_regions = []
        for region in regions:
            if len(region.shape) == 3:
                region_tensor = self.transform(region)
                batch_regions.append(region_tensor)

        if len(batch_regions) == 0:
            return torch.empty(0, 4096)

        # Stack into batch
        batch_tensor = torch.stack(batch_regions).to(self.device)

        # Extract features
        with torch.no_grad():
            features = self.feature_extractor(batch_tensor)

        return features

    @staticmethod
    def calculate_iou(box1: BoundingBox, box2: BoundingBox) -> float:
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        # Calculate intersection
        xl = max(x1, x2)
        xr = min(x1 + w1, x2 + w2)
        yt = max(y1, y2)
        yb = min(y1 + h1, y2 + h2)

        if xr <= xl or yb <= yt:
            return 0.0

        intersection = (xr - xl) * (yb - yt)
        union = (w1 * h1) + (w2 * h2) - intersection

        return intersection / union if union > 0 else 0.0

    def prepare_training(
            self,
            images: List[NDArray],
            ground_truth_boxes: List[List[BoundingBox]],
            ground_truth_labels: List[List[int]],
    ) -> Tuple[List[NDArray], List[int]]:
        all_regions = []
        all_labels = []

        for idx, image in enumerate(images):
            # Get region proposals using selective search
            proposals = self.selective_search.proposals(image)  # 2000k region proposals
            gt_boxes = ground_truth_boxes[idx]
            gt_labels = ground_truth_labels[idx]

            # Loop through each proposal and check against ground truth boxes
            for proposal in proposals:
                x, y, w, h = proposal
                region = image[y:y + h, x:x + w]
                if region.size == 0:
                    continue

                # Check IoU with ground truth boxes
                max_iou = 0.0
                best_label = 0  # Background

                for gt_box, gt_label in zip(gt_boxes, gt_labels):
                    iou = self.calculate_iou(proposal, gt_box)
                    if iou > max_iou:
                        max_iou = iou
                        if iou >= 0.5:  # Positive sample threshold
                            best_label = gt_label

                # Find the best label based on IoU
                all_regions.append(region)
                all_labels.append(best_label)

        return all_regions, all_labels

    def train(
            self,
            images: List[NDArray],
            ground_truth_boxes: List[List[BoundingBox]],
            ground_truth_labels: List[List[int]],
            epochs: int = 10,
            batch_size: int = 32,
            learning_rate: float = 0.001,
    ) -> None:
        # Prepare training data
        print("Starting preparation of training data 🫧")
        regions, labels = self.prepare_training(images, ground_truth_boxes, ground_truth_labels)
        print(f"Prepared {len(regions)} regions for training.")

        # Extract features from regions
        # features = self.extract_features(regions)
        # print(f"Extracted features shape: {features.shape}")
