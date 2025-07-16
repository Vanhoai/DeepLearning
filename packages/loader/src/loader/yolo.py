import os
import cv2

class CustomYOLODataset:
    def __init__(self, video_path=None, dataset_directory=None):
        """
        Initialize the CustomYOLODataset with paths to the video file and dataset directory.
        Args:
            video_path (str): Path to the video file from which images will be extracted.
            dataset_directory (str): Directory where the images and annotations will be saved.

        Example:
            if you want to create a traffic dataset from a video file path/to/traffic.mp4 and
            save in data/traffic:
            dataset = CustomYOLODataset(video_path="path/to/traffic.mp4", dataset_directory="path/to/data/traffic")

        Note:
            Dataset directory should contain 'images/train', 'images/val' folders following YOLO format.
            <name-dataset>
            - <name-dataset>.yaml
            - images
                - train
                    - image000.jpg
                    - image001.jpg
                    .....
                - val
                    - image000.jpg
                    - image001.jpg
                    .....
            - labels
                - train
                    - image000.txt
                    - image001.txt
                    .....
                - val
                    - image000.txt
                    - image001.txt
                    .....
        """
        if video_path is None or not os.path.exists(video_path):
            raise ValueError("Please provide path to video file for generating dataset")
        
        if dataset_directory is None or not os.path.exists(dataset_directory):
            raise ValueError("Please provide path to dataset directory for saving images and annotations")

        self.root_directory = os.getcwd()
        self.video_path = video_path
        self.dataset_directory = dataset_directory

        # init format directory structure
        self.init_format()

    def init_format(self):
        images_directory = os.path.join(self.dataset_directory, "images")
        labels_directory = os.path.join(self.dataset_directory, "labels")

        if not os.path.exists(images_directory):
            os.makedirs(images_directory)

        if not os.path.exists(labels_directory):
            os.makedirs(labels_directory)

        self.image_train_directory = os.path.join(images_directory, "train")
        self.image_val_directory = os.path.join(images_directory, "val")
        self.label_train_directory = os.path.join(labels_directory, "train")
        self.label_val_directory = os.path.join(labels_directory, "val")

        os.makedirs(self.image_train_directory, exist_ok=True)
        os.makedirs(self.image_val_directory, exist_ok=True)

        os.makedirs(self.label_train_directory, exist_ok=True)
        os.makedirs(self.label_val_directory, exist_ok=True)
        
    def generate(self):
        result = self.write_images() and self.write_labels()
        print("Dataset generation completed ✅." if result else "Dataset generation failed 😞.")

    def write_images(self) -> bool:
        video = cv2.VideoCapture(self.video_path)
        if not video.isOpened():
            raise ValueError(f"Could not open video file: {self.video_path}")
        
        FRAME_COUNT = int(video.get(cv2.CAP_PROP_FRAME_COUNT)) # 1242
        FRAME_STEP = FRAME_COUNT // 4  # Step to get 4 evenly spaced frames
        
        # write 4 images to folder train
        for i in range(4):
            frame_number = i * FRAME_STEP
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = video.read()
            if not ret:
                raise ValueError(f"Could not read frame {frame_number} from video.")
            
            image_path = os.path.join(self.image_train_directory, f"frame_000{i + 1}.jpg")
            cv2.imwrite(image_path, frame)
            print(f"Saved train frame_000{i + 1} to {image_path}")

        # write 3 image to folder val
        idx =[0, FRAME_COUNT / 2, FRAME_COUNT - 1]
        for i in range(len(idx)):
            frame_number = int(idx[i])

            video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = video.read()
            if not ret:
                raise ValueError(f"Could not read frame {idx} from video.")
            
            image_path = os.path.join(self.image_val_directory, f"frame_000{i + 1}.jpg")
            cv2.imwrite(image_path, frame)
            print(f"Saved val frame_000{i + 1} to {image_path}")

        return True

    def write_labels(self)-> bool:
        # This method should implement the logic to write labels for the images
        # For now, it is a placeholder
        return True