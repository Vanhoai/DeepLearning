from models import RCNN
from loader import load_pascal_voc_dataset
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random

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


def main() -> None:
    rcnn = RCNN(num_classes=21)
    images, boxes, labels = load_pascal_voc_dataset(path, split="train")

    n = 500

    images = images[:n]
    rcnn.train(images, boxes, labels, epochs=5, batch_size=256, learning_rate=0.001)
