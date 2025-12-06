import torch
from torch.utils.data import Dataset


class PointsDataset(Dataset):
    def __init__(
        self,
        num_classes: int = 4,
        num_points_per_class: int = 1000,
        mean: torch.Tensor = torch.tensor(
            [
                [0.0, 0.0],
                [5.0, 5.0],
                [0.0, 5.0],
                [5.0, 0.0],
            ]
        ),
        cov: torch.Tensor = torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
    ):
        self.num_classes = num_classes
        self.length = num_classes * num_points_per_class

        # (N, 2)
        self.points: torch.Tensor = torch.zeros((num_classes * num_points_per_class, 2))
        self.labels: torch.Tensor = torch.zeros(
            (num_classes * num_points_per_class,),
            dtype=torch.long,
        )

        for class_idx in range(num_classes):
            mean_vector = mean[class_idx]
            class_points = torch.distributions.MultivariateNormal(
                mean_vector,
                cov,
            ).sample((num_points_per_class,))

            start_idx = class_idx * num_points_per_class
            end_idx = start_idx + num_points_per_class

            self.points[start_idx:end_idx, :] = class_points
            self.labels[start_idx:end_idx] = class_idx

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
        point = self.points[idx]
        label = self.labels[idx]

        return point, label
