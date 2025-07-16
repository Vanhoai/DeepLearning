import os
from loader import CustomYOLODataset

root = os.getcwd()
video_path = os.path.join(root, "assets", "traffic.mp4")
dataset_directory = os.path.join(root, "data", "traffic")

def main() -> None:
    dataset = CustomYOLODataset(video_path, dataset_directory)
    dataset.generate()
    