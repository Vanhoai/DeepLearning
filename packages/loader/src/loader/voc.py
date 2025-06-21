import os
import cv2
from xml.etree import ElementTree


def load_pascal_voc_dataset(dataset_path: str, split: str = 'train'):
    """
    Load PASCAL VOC dataset

    Args:
        dataset_path: Path to VOC dataset (e.g., '/path/to/VOCdevkit/VOC2012')
        split: 'train', 'val', or 'test'

    Returns:
        images, boxes, labels
    """

    # VOC class names
    classes = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle",
               "bus", "car", "cat", "chair", "cow", "diningtable", "dog",
               "horse", "motorbike", "person", "pottedplant", "sheep",
               "sofa", "train", "tvmonitor"]

    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

    # Paths
    image_dir = os.path.join(dataset_path, "JPEGImages")
    annotation_dir = os.path.join(dataset_path, "Annotations")
    split_file = os.path.join(dataset_path, "ImageSets", "Main", f"{split}.txt")

    if not os.path.exists(split_file):
        print(f"Split file not found: {split_file}")
        return [], [], []

    # Read image IDs
    with open(split_file, "r") as f:
        image_ids = [line.strip() for line in f.readlines()]

    images = []
    all_boxes = []
    all_labels = []

    for img_id in image_ids:
        # Load image
        img_path = os.path.join(image_dir, f"{img_id}.jpg")
        if not os.path.exists(img_path):
            continue

        image = cv2.imread(img_path)
        if image is None:
            continue

        # Load annotation
        ann_path = os.path.join(annotation_dir, f"{img_id}.xml")
        if not os.path.exists(ann_path):
            continue

        tree = ElementTree.parse(ann_path)
        root = tree.getroot()

        boxes = []
        labels = []

        for obj in root.findall("object"):
            # Get class name
            class_name = obj.find("name").text
            if class_name not in class_to_idx:
                continue

            # Get bounding box
            bbox = obj.find("bndbox")
            xmin = int(bbox.find("xmin").text)
            ymin = int(bbox.find("ymin").text)
            xmax = int(bbox.find("xmax").text)
            ymax = int(bbox.find("ymax").text)

            # Convert to (x, y, w, h) format
            w = xmax - xmin
            h = ymax - ymin

            boxes.append((xmin, ymin, w, h))
            labels.append(class_to_idx[class_name])

        if len(boxes) > 0:  # Only add images with annotations
            images.append(image)
            all_boxes.append(boxes)
            all_labels.append(labels)

    print(f"Loaded {len(images)} images from {split} split")
    return images, all_boxes, all_labels
