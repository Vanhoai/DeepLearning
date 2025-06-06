from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, List
from numpy.typing import NDArray
import numpy as np

from src.nn.layer import Layer
from src.nn.loss import Loss
from src.nn.optimizer import Optimizer


class Model(ABC):
    @abstractmethod
    def feedforward(self, X: NDArray) -> NDArray: ...

    @abstractmethod
    def backpropagation(self, Y: NDArray): ...

    @abstractmethod
    def predict(self, X: NDArray) -> NDArray: ...

    @abstractmethod
    def evaluate(self, Y: NDArray, A: NDArray) -> NDArray: ...

    @abstractmethod
    def update(self) -> None: ...

    @abstractmethod
    def fit(
        self,
        X: NDArray,
        Y: NDArray,
        epochs: int = 100,
        batch_size: int = 32,
        validation_data: Optional[Tuple[NDArray, NDArray]] = None,
    ) -> Dict[str, List[Any]]: ...


class Sequential(Model):
    """
    X: R(N X D)
    """

    def __init__(
        self,
        layers: List[Layer],
        loss: Loss,
        optimizer: Optimizer,
        regularization: float = 0.0,
    ) -> None:
        self.layers = layers
        self.loss = loss
        self.optimizer = optimizer
        self.regularization = regularization

        # Attributes will be update later
        self.N: Optional[int] = None
        self.input: Optional[NDArray] = None
        self.output: Optional[NDArray] = None

    def feedforward(self, X: NDArray) -> NDArray:
        O = X
        for L in self.layers:
            O = L.forward(O)

        return O

    def backpropagation(self, Y: NDArray, batch_size: int = 32) -> None:
        assert self.output is not None, "Output must be computed before backpropagation"

        A = self.output  # output of the last layer
        dA = self.loss.derivative(Y, A)
        # gradient of the loss with respect to the output
        # layers = [ReLu(d, 512), ReLu(512, 128), ReLu(128, 64), Softmax(64, classes)]

        dW = [None] * len(self.layers)
        db = [None] * len(self.layers)

        for index in reversed(range(len(self.layers))):
            assert self.layers[index].Z is not None

            # if activation is None, derivativeZ will return 1, so it not affects the gradient
            dZ = dA * self.layers[index].activation.derivative(self.layers[index].Z)  # type: ignore[assignment]

            AP = None  # Post-activation output of the previous layer
            if index > 0:
                AP = self.layers[index - 1].A
            else:
                AP = self.input

            assert AP is not None, "AP must be computed before backpropagation"
            dW[index] = AP.T @ dZ / batch_size  # type: ignore[broadcasting numpy]
            db[index] = np.mean(dZ, axis=0, keepdims=True)

            if index > 0:
                dA = dZ @ self.layers[index].W.T

        eta = 1e-2  # Learning rate, can be set as a parameter
        for i in range(len(self.layers)):
            self.layers[i].W -= eta * dW[i]  # type: ignore
            self.layers[i].b -= eta * db[i]  # type: ignore

    def predict(self, X: NDArray) -> NDArray:
        return self.feedforward(X)

    # FIXME: if A and Y are not one-hot encoded, this function will not work correctly
    def evaluate(self, Y: NDArray, A: NDArray) -> NDArray:
        # d: dimensions of A and Y must be the same
        # A: R(N x d), Y: R(N x d)
        Y = np.argmax(Y, axis=1)
        A = np.argmax(A, axis=1)
        return np.mean(Y == A)

    def update(self) -> None:
        for layer in self.layers:
            # Apply regularization
            # if self.regularization > 0:
            #     layer.dW += self.regularization * layer.W
            # Update parameters using optimizer
            assert layer.dW is not None, "dW must be computed before update"
            assert layer.db is not None, "db must be computed before update"
            self.optimizer.update(layer, layer.dW, layer.db)

    def fit(
        self,
        X: NDArray,
        Y: NDArray,
        epochs: int = 50,
        batch_size: int = 32,
        validation_data: Optional[Tuple[NDArray, NDArray]] = None,
    ) -> Dict[str, List[Any]]:
        self.N = X.shape[0]
        for epoch in range(epochs):
            # Shuffle the datasets
            indices = np.arange(X.shape[0])
            np.random.shuffle(indices)
            X_shuffle = X[indices]
            Y_shuffle = Y[indices]

            # Iterate over batches
            for start_batch in range(0, self.N, batch_size):
                end_batch = min(start_batch + batch_size, self.N)
                X_batch = X_shuffle[start_batch:end_batch]
                Y_batch = Y_shuffle[start_batch:end_batch]

                # Feedforward & Backpropagation
                self.input = X_batch
                output = self.feedforward(X_batch)
                self.output = output  # Store the output for backpropagation
                self.backpropagation(Y_batch, batch_size)
                # Update parameters of model
                # self.update()

            # Calculate loss and accuracy for the current epoch
            YA = self.predict(X)
            loss = self.loss(Y, YA)
            accuracy = self.evaluate(Y, YA)
            msg = f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}"
            print(msg)

        # return history of loss and accuracy of training and validation
        history: Dict[str, List[Any]] = {
            "loss": [],
            "accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
        }

        return history
