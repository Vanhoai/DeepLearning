import cv2


def load_coco_dataset(dataset_path: str, split: str = "train2017"):
    """
    Load COCO dataset

    Args:
        dataset_path: Path to COCO dataset (e.g., '/path/to/coco')
        split: 'train2017', 'val2017', etc.

    Returns:
        images, boxes, labels
    """
    try:
        from pycocotools.coco import COCO
    except ImportError:
        print("Please install pycocotools: pip install pycocotools")
        return [], [], []

    import os

    # Paths
    image_dir = os.path.join(dataset_path, split)
    ann_file = os.path.join(dataset_path, 'annotations', f'instances_{split}.json')

    if not os.path.exists(ann_file):
        print(f"Annotation file not found: {ann_file}")
        return [], [], []

    # Initialize COCO api
    coco = COCO(ann_file)

    # Get all image IDs
    img_ids = coco.getImgIds()

    images = []
    all_boxes = []
    all_labels = []

    for img_id in img_ids[:1000]:  # Limit to 1000 images for demo
        # Load image info
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(image_dir, img_info['file_name'])

        if not os.path.exists(img_path):
            continue

        image = cv2.imread(img_path)
        if image is None:
            continue

        # Get annotations
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        boxes = []
        labels = []

        for ann in anns:
            if ann['iscrowd']:
                continue

            # Get bounding box (COCO format: [x, y, width, height])
            bbox = ann['bbox']
            x, y, w, h = bbox

            # Filter out very small boxes
            if w < 10 or h < 10:
                continue

            boxes.append((int(x), int(y), int(w), int(h)))
            labels.append(ann['category_id'])

        if len(boxes) > 0:
            images.append(image)
            all_boxes.append(boxes)
            all_labels.append(labels)

    print(f"Loaded {len(images)} images from COCO {split}")
    return images, all_boxes, all_labels
