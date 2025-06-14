import numpy as np
from typing import Tuple, List
from abc import ABC, abstractmethod
from numpy.typing import NDArray


class ActivationFunction(ABC):
    @abstractmethod
    def __call__(self, x: NDArray) -> NDArray: ...

    @abstractmethod
    def gradient(self, x: NDArray) -> NDArray: ...


class LossFunction(ABC):
    @abstractmethod
    def __call__(self, Y_true: NDArray, Y_pred: NDArray) -> float: ...

    @abstractmethod
    def gradient(self, Y_true: NDArray, Y_pred: NDArray) -> NDArray: ...


class CrossEntropyLoss(LossFunction):
    def __call__(self, Y_true: NDArray, Y_pred: NDArray) -> float:
        epsilon = 1e-15
        Y_pred = np.clip(Y_pred, epsilon, 1 - epsilon)
        return -np.mean(np.sum(Y_true * np.log(Y_pred), axis=1))

    def gradient(self, Y_true: NDArray, Y_pred: NDArray) -> NDArray:
        return Y_pred - Y_true


class ReLU(ActivationFunction):
    def __call__(self, x: NDArray) -> NDArray:
        return np.maximum(0, x)

    def gradient(self, x: NDArray) -> NDArray:
        return np.where(x > 0, 1, 0)


class Softmax(ActivationFunction):
    def __call__(self, Z: NDArray) -> NDArray:
        exp = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return exp / exp.sum(axis=1, keepdims=True)

    def gradient(self, x: NDArray) -> NDArray:
        return np.ones_like(x)


class NeuralNetworkV2:
    """
    Initializes a modern neural network with specified layers and learning rate.
    Parameters:
    @param layers: List layers, with each layer consist of number of units and activation function.
                   Example: [(784, ReLU()), (128, ReLU()), (10, Softmax())]
    @param eta: Float representing the learning rate for the optimizer.

    Example:
    >>> nn = ModernNeuralNetwork(layers=[(784, ReLU()), (128, ReLU()), (10, Softmax())], loss= CrossEntropyLoss(), eta=0.001)
    """

    def __init__(
        self,
        layers: List[Tuple[int, ActivationFunction]],
        loss: LossFunction,
        eta=1e-3,
    ):
        self.eta: float = eta
        self.layers: List[Tuple[int, ActivationFunction]] = layers
        self.loss: LossFunction = loss

        # Properties to hold weights, biases, and activations
        self.W: List[NDArray] = []
        self.b: List[NDArray] = []
        self.Z: List[NDArray] = []

        # Initialize weights and biases
        for i in range(len(layers) - 1):
            di = layers[i][0]
            do = layers[i + 1][0]

            W_ = (np.random.randn(di, do) * np.sqrt(2.0 / di)).astype(np.float32)
            b_ = np.zeros((1, do), dtype=np.float32)

            self.W.append(W_)
            self.b.append(b_)

    def feed_forward(self, X: NDArray) -> Tuple[List[NDArray], List[NDArray]]:
        A = [X]
        Z = []

        O = A[-1]
        # Feedforward
        for i in range(len(self.layers) - 1):
            Z_ = O @ self.W[i] + self.b[i]  # Z[i+1] = pre-activation of layer i + 1
            Z.append(Z_)
            O = self.layers[i + 1][1](Z_)  # A[i+1] = activation of layer i + 1
            A.append(O)

        return A, Z

    def backpropagation(self, A: List[NDArray], Z: List[NDArray], Y: NDArray):
        N = Y.shape[0]
        L = len(self.layers) - 1

        # Initialize gradients
        dW = [None] * L
        db = [None] * L

        # Backpropagation
        dA = self.loss.gradient(Y, A[-1])

        for i in reversed(range(len(self.layers) - 1)):
            dZ = dA * self.layers[i + 1][1].gradient(Z[i])
            dW[i] = A[i].T @ dZ / N  # type: ignore
            db[i] = np.mean(dZ, axis=0, keepdims=True)

            # Prepare for the previous layer
            if i > 0:
                dA = dZ @ self.W[i].T

        return dW, db

    def predict(self, X: NDArray) -> NDArray:
        A, _ = self.feed_forward(X)
        return A[-1]  # Return the output of the last layer

    @staticmethod
    def compute_accuracy(Y_true: NDArray, Y_pred: NDArray):
        """
        Computes the accuracy of predictions.
        :param Y_true: True labels (one-hot encoded).
        :param Y_pred: Predicted labels (one-hot encoded).
        :return: Accuracy as a float.
        """
        y_true = np.argmax(Y_true, axis=1)
        y_pred = np.argmax(Y_pred, axis=1)
        return np.mean(y_true == y_pred)

    def fit(
        self,
        X: NDArray,
        Y: NDArray,
        epochs: int = 40,
        verbose: bool = True,
        frequency: int = 10,
        batch_size: int = 32,
    ) -> dict[str, List[float]]:
        history: dict = {
            "loss": [],
            "accuracy": [],
        }

        for epoch in range(epochs):
            N = X.shape[0]
            indices = np.arange(N)
            np.random.shuffle(indices)
            X_shuffled = X[indices]
            Y_shuffled = Y[indices]

            for start_idx in range(0, N, batch_size):
                end_idx = min(start_idx + batch_size, N)
                X_batch = X_shuffled[start_idx:end_idx]
                Y_batch = Y_shuffled[start_idx:end_idx]

                # Feedforward & Backpropagation in batch
                A_, Z_ = self.feed_forward(X_batch)
                dW, db = self.backpropagation(A_, Z_, Y_batch)

                # Update weights and biases
                for i in range(len(self.layers) - 1):
                    self.W[i] -= self.eta * dW[i]  # type: ignore
                    self.b[i] -= self.eta * db[i]  # type: ignore

            # Logging after each epoch
            if verbose and (epoch % frequency == 0):
                AF, _ = self.feed_forward(X)
                loss = self.loss(Y, AF[-1])
                accuracy = self.compute_accuracy(Y, AF[-1])
                history["loss"].append(loss)
                history["accuracy"].append(accuracy)

                print(f"epoch: {epoch}, loss: {loss:.4f}, accuracy: {accuracy:.4f}")

        return history
