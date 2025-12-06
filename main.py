# type: ignore
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from foundations import (
    HSAdam,
    HSConv2D,
    HSCrossEntropyLoss,
    HSDropout,
    HSFlatten,
    HSLinear,
    HSMaxPool2d,
    HSModule,
    HSReLU,
)


class MLP(HSModule):
    def __init__(
        self,
        in_features: int = 784,
        classes: int = 10,
    ):
        super().__init__()
        # [B, 1, 28, 28] -> [B, 784]
        self.flatten = HSFlatten()

        # [B, 784] -> [B, 256]
        self.fc1 = HSLinear(in_features=in_features, out_features=256, bias=True)
        self.relu1 = HSReLU()
        self.dropout1 = HSDropout(0.3)

        # [B, 256] -> [B, 128]
        self.fc2 = HSLinear(in_features=256, out_features=128, bias=True)
        self.relu2 = HSReLU()
        self.dropout2 = HSDropout(0.3)

        # [B, 128] -> [B, 10]
        self.fc3 = HSLinear(in_features=128, out_features=classes, bias=True)

        # Params:
        # FC1: (256 * 28 * 28) + 256 = 200.960
        # FC2: (128 * 256) + 128 = 32.896
        # FC3: (10 * 128) + 10 = 1.290
        # Total: 235.146

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self.flatten(X)
        X = self.fc1(X)
        X = self.relu1(X)
        X = self.dropout1(X)

        X = self.fc2(X)
        X = self.relu2(X)
        X = self.dropout2(X)

        X = self.fc3(X)
        return X

    def backward(self, grad_output: torch.Tensor) -> torch.Tensor:
        grad = self.fc3.backward(grad_output)

        grad = self.dropout2.backward(grad)
        grad = self.relu2.backward(grad)
        grad = self.fc2.backward(grad)

        grad = self.dropout1.backward(grad)
        grad = self.relu1.backward(grad)
        grad = self.fc1.backward(grad)
        grad = self.flatten.backward(grad)

        return grad


class CNN(HSModule):
    def __init__(self):
        super().__init__()
        # Convolutional Layers

        # [B, 1, 28, 28] -> [B, 32, 14, 14]
        self.conv1 = HSConv2D(1, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = HSReLU()
        self.pool1 = HSMaxPool2d(kernel_size=2)

        # [B, 32, 14, 14] -> [B, 64, 7, 7]
        self.conv2 = HSConv2D(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = HSReLU()
        self.pool2 = HSMaxPool2d(kernel_size=2)

        # FC Layers
        # [B, 64, 7, 7] -> [B, 64 * 7 * 7]
        self.flatten = HSFlatten()
        # [B, 64 * 7 * 7] -> [B, 128]
        self.fc1 = HSLinear(64 * 7 * 7, 128)
        self.relu3 = HSReLU()
        self.dropout = HSDropout(0.5)
        self.fc2 = HSLinear(128, 10)  # [B, 10]

        # Params:
        # Conv1: (32 * 1 * 3 * 3) + 32 = 320
        # Conv2: (64 * 32 * 3 * 3) + 64 = 18.496
        # FC1: (128 * 64 * 7 * 7) + 128 = 401.536
        # FC2: (10 * 128) + 10 = 1.290
        # Total: 421.642

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self.conv1(X)
        X = self.relu1(X)
        X = self.pool1(X)

        X = self.conv2(X)
        X = self.relu2(X)
        X = self.pool2(X)

        X = self.flatten(X)
        X = self.fc1(X)
        X = self.relu3(X)
        X = self.dropout(X)
        X = self.fc2(X)

        return X

    def backward(self, grad_output: torch.Tensor) -> torch.Tensor:
        grad = self.fc2.backward(grad_output)
        grad = self.dropout.backward(grad)
        grad = self.relu3.backward(grad)
        grad = self.fc1.backward(grad)
        grad = self.flatten.backward(grad)

        grad = self.pool2.backward(grad)
        grad = self.relu2.backward(grad)
        grad = self.conv2.backward(grad)

        grad = self.pool1.backward(grad)
        grad = self.relu1.backward(grad)
        grad = self.conv1.backward(grad)

        return grad


def train_mnist():
    BATCH_SIZE = 100
    CLASSES = 10
    LR = 0.0001
    EPOCHS = 20
    DEVICE = "mps" if torch.mps.is_available() else "cpu"

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    train_dataset = datasets.MNIST(
        root="./data",
        train=True,
        transform=transform,
    )

    val_dataset = datasets.MNIST(
        root="./data",
        train=False,
        transform=transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=8,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=8,
    )

    model = CNN()
    criterion = HSCrossEntropyLoss()
    optimizer = HSAdam(model.parameters(), learning_rate=LR)

    for epoch in range(EPOCHS):
        # Traing Phase
        model.train()
        train_loss = 0.0

        train_progress = tqdm(train_loader, colour="blue")
        for batch_idx, (images, targets) in enumerate(train_progress):
            # Forward
            outputs = model.forward(images)
            loss = criterion.forward(outputs, targets)

            # Backward
            optimizer.zero_grad()
            grad = criterion.backward(outputs)
            model.backward(grad)

            # Update
            optimizer.step()

            train_loss += loss.item()
            msg = f"Epoch [{epoch + 1}/{EPOCHS}], Step [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}"
            train_progress.set_description(msg)


if __name__ == "__main__":
    train_mnist()
