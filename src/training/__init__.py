from loader import load_pascal_voc_dataset
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
from typing import Tuple, List
from numpy.typing import NDArray
from models import SelectiveSearch
import multiprocessing

path = "/Users/hinsun/Workspace/ComputerScience/DeepLearning/data/VOCdevkit/VOC2012"
voc_classes = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]


def show_samples(images, boxes, labels, num_samples=10):
    indices = random.sample(range(len(images)), num_samples)

    r, c = 2, 5
    fig, axes = plt.subplots(r, c, figsize=(15, 6))

    for i, idx in enumerate(indices):
        ax = axes[i // c, i % c]
        ax.imshow(images[idx])
        ax.axis("off")

        for box, label in zip(boxes[idx], labels[idx]):
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1

            rect = patches.Rectangle(
                (x1, y1), width, height, linewidth=2, edgecolor="red", facecolor="none"
            )
            ax.add_patch(rect)
            ax.text(
                x1,
                y1 - 5,
                voc_classes[label],
                color="white",
                fontsize=12,
                bbox=dict(facecolor="red", alpha=0.5),
            )

    plt.tight_layout()
    plt.show()


type BoundingBox = Tuple[int, int, int, int]  # (x, y, width, height)


class RCNN:
    def __init__(self, num_classes=21):  # 20 + 1 for background
        self.num_classes = num_classes
        self.selective_search = SelectiveSearch()

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

    def process_image(self, args):
        image, gt_boxes, gt_labels = args

        proposals = self.selective_search.proposals(image)
        regions = []
        labels = []

        for proposal in proposals:
            x, y, w, h = proposal
            region = image[y : y + h, x : x + w]
            if region.size == 0:
                continue

            max_iou = 0.0
            best_label = 0  # background
            for gt_box, gt_label in zip(gt_boxes, gt_labels):
                iou = RCNN.calculate_iou(gt_box, proposal)
                if iou > max_iou:
                    max_iou = iou
                    if iou >= 0.5:
                        best_label = gt_label

            regions.append(region)
            labels.append(best_label)

        return regions, labels

    def prepare_training(
        self,
        images: List[NDArray],
        ground_truth_boxes: List[List[BoundingBox]],
        ground_truth_labels: List[List[int]],
    ) -> Tuple[List[NDArray], List[int]]:
        # all_regions = []
        # all_labels = []

        # for idx, image in enumerate(images):
        #     # Get region proposals using selective search
        #     proposals = self.selective_search.proposals(image)  # 2000k region proposals
        #     gt_boxes = ground_truth_boxes[idx]
        #     gt_labels = ground_truth_labels[idx]

        #     # Loop through each proposal and check against ground truth boxes
        #     for proposal in proposals:
        #         x, y, w, h = proposal
        #         region = image[y : y + h, x : x + w]
        #         if region.size == 0:
        #             continue

        #         # Check IoU with ground truth boxes
        #         max_iou = 0.0
        #         best_label = 0  # Background

        #         for gt_box, gt_label in zip(gt_boxes, gt_labels):
        #             iou = self.calculate_iou(proposal, gt_box)
        #             if iou > max_iou:
        #                 max_iou = iou
        #                 if iou >= 0.5:  # Positive sample threshold
        #                     best_label = gt_label

        #         # Find the best label based on IoU
        #         all_regions.append(region)
        #         all_labels.append(best_label)

        # return all_regions, all_labels

        pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
        print(f"Using {multiprocessing.cpu_count()} parallel processes 🧠")

        tasks = [
            (
                images[i],
                ground_truth_boxes[i],
                ground_truth_labels[i],
            )
            for i in range(len(images))
        ]

        results = pool.map(self.process_image, tasks)
        pool.close()
        pool.join()

        all_regions = []
        all_labels = []

        for regions, labels in results:
            all_regions.extend(regions)
            all_labels.extend(labels)

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
        regions, labels = self.prepare_training(
            images, ground_truth_boxes, ground_truth_labels
        )
        print(f"Prepared {len(regions)} regions for training.")

        # Extract features from regions
        # features = self.extract_features(regions)
        # print(f"Extracted features shape: {features.shape}")


def main() -> None:
    rcnn = RCNN(num_classes=21)
    images, boxes, labels = load_pascal_voc_dataset(path, split="train")
    rcnn.train(images, boxes, labels, epochs=10, batch_size=32, learning_rate=0.001)
