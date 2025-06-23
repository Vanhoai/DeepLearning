from typing import List, Tuple
import cv2
from numpy.typing import NDArray


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
