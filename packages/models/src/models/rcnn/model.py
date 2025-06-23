import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from typing import List, Tuple
from numpy.typing import NDArray
from .dataset import RCNNDataset, RegionsDataset
from .search import SelectiveSearch
from .features import FeatureExtractor
from .classifier import SoftmaxClassifier

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
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Set eval mode for feature extractor avoid gradient computation
        self.feature_extractor.eval()

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

    def extract_features(self, regions: torch.Tensor) -> torch.Tensor:
        if len(regions) == 0:
            return torch.empty(0, 4096)

        # Extract features
        with torch.no_grad():
            features = self.feature_extractor(regions)

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
            print(f"Processing image {idx + 1}/{len(images)}: {len(proposals)} proposals found")

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
        print("Number of regions prepared for training:", len(regions))

        # Build dataset for RCNN
        dataset = RegionsDataset(regions=regions, labels=labels, transform=self.transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Setup optimizer
        optimizer = optim.Adam(self.classifier.parameters(), lr=learning_rate)

        # Training loop
        self.feature_extractor.eval()  # Keep feature extractor frozen
        self.classifier.train()

        for epoch in range(epochs):
            total_loss = 0.0
            correct = 0
            total = 0

            for batch_idx, item in enumerate(dataloader):
                regions_batch, labels_batch = item

                # convert to tensors and move to device
                regions_batch = regions_batch.to(self.device)  # type: ignore
                labels_batch = torch.tensor(labels_batch, dtype=torch.long).to(self.device)

                # Extract features for the batch
                features = self.extract_features(regions_batch)  # type: ignore
                if features.size(0) == 0:
                    continue

                # output of feature extractor is (batch_size, 4096)
                # Forward pass
                optimizer.zero_grad()
                outputs = self.classifier(features)
                loss = self.criterion(outputs, labels_batch)

                # Backward pass
                loss.backward()
                optimizer.step()

                # Statistics
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels_batch.size(0)
                correct += (predicted == labels_batch).sum().item()  # type: ignore

                if batch_idx % 10 == 0:
                    print(f'Epoch [{epoch + 1}/{epochs}], Batch [{batch_idx + 1}], '
                          f'Loss: {loss.item():.4f}')

            accuracy = 100 * correct / total if total > 0 else 0
            avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
            print(f'Epoch [{epoch + 1}/{epochs}] completed: '
                  f'Average Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')

        print("Training completed!")
