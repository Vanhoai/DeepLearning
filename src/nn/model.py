from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, List
from numpy.typing import NDArray
import numpy as np
import os

from src.nn.layer import Layer
from src.nn.loss import Loss
from src.nn.optimizer import Optimizer
from src.nn.early_stopping import EarlyStopping
from src.nn.regularization import RegularizationType, RegularizationFactory, Regularization


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
    def update_parameters(self) -> None: ...

    @abstractmethod
    def fit(
            self,
            X: NDArray,
            Y: NDArray,
            epochs: int = 100,
            batch_size: int = 32,
            validation_data: Optional[Tuple[NDArray, NDArray]] = None,
    ) -> Dict[str, List[Any]]: ...

    @abstractmethod
    def load(self, path: str) -> None: ...

    @abstractmethod
    def save(self, path: str) -> None: ...


class Sequential(Model):
    def __init__(
            self,
            layers: List[Layer],
            loss: Loss,
            optimizer: Optimizer,
            regularization: Optional[RegularizationType] = None,
            regularization_lambda: float = 1e-2,
    ) -> None:
        self.layers = layers
        self.loss = loss
        self.optimizer = optimizer
        self.regularization: Optional[Regularization] = None

        # Regularization
        if regularization is not None:
            self.regularization = RegularizationFactory.create(regularization, regularization_lambda)
        else:
            self.regularization = None

        # Attributes will be updated later
        self.N: Optional[int] = None
        self.input: Optional[NDArray] = None
        self.output: Optional[NDArray] = None
        self.batch_size: int = 32

    def feedforward(self, X: NDArray) -> NDArray:
        self.input = X

        O = X
        for L in self.layers:
            O = L.forward(O)

        self.output = O
        return O

    def backpropagation(self, Y: NDArray) -> None:
        assert self.output is not None, "Output must be computed before backpropagation"

        A = self.output  # output of the last layer
        dA = self.loss.derivative(Y, A)

        for index in reversed(range(len(self.layers))):
            assert self.layers[index].Z is not None

            layer = self.layers[index]

            # if activation is None, derivativeZ will return 1, so it not affects the gradient
            dA_dZ = layer.activation.derivative(layer.Z)  # type: ignore
            dZ = dA * dA_dZ

            AP = None  # Post-activation output of the previous layer
            if index > 0:
                AP = self.layers[index - 1].A
            else:
                AP = self.input

            assert AP is not None, "AP must be computed before backpropagation"
            layer.dW = AP.T @ dZ / self.batch_size  # type: ignore[broadcasting numpy]
            layer.db = np.mean(dZ, axis=0, keepdims=True)

            # Regularization
            if self.regularization is not None:
                rdW = self.regularization.compute_gradient(layer.W)
                layer.dW += rdW

            if index > 0:
                dA = dZ @ layer.W.T

    def predict(self, X: NDArray) -> NDArray:
        return self.feedforward(X)

    # FIXME: if A and Y are not one-hot encoded, this function will not work correctly
    def evaluate(self, Y: NDArray, A: NDArray):
        # d: dimensions of A and Y must be the same
        # A: R(N x d), Y: R(N x d)
        Y = np.argmax(Y, axis=1)
        A = np.argmax(A, axis=1)
        return np.mean(Y == A)

    def update_parameters(self) -> None:
        for layer in self.layers:
            # Apply regularization
            # if self.regularization > 0:
            #     layer.dW += self.regularization * layer.W
            # Update parameters using optimizer
            assert layer.dW is not None, "dW must be computed before update"
            assert layer.db is not None, "db must be computed before update"
            # self.optimizer.update(layer, layer.dW, layer.db)

            # Update the parameters of the model
            eta = self.optimizer.eta
            layer.W -= eta * layer.dW  # type: ignore
            layer.b -= eta * layer.db  # type: ignore

    @staticmethod
    def prepare_data(
            X: NDArray,
            Y: NDArray,
            validation_data: Optional[Tuple[NDArray, NDArray]] = None,
    ):
        # prepare validation data, if not provided => split 20% of the training data
        if validation_data is None:
            split_index = int(0.8 * X.shape[0])
            X_train, Y_train = X[:split_index], Y[:split_index]
            X_val, Y_val = X[split_index:], Y[split_index:]
        else:
            X_train, Y_train = X, Y
            X_val, Y_val = validation_data

        return X_train, Y_train, X_val, Y_val

    def calculate_loss_accuracy(self, X: NDArray, Y: NDArray):
        YA = self.predict(X)
        base_loss = self.loss(Y, YA)
        accuracy = self.evaluate(Y, YA)

        if self.regularization is not None:
            # Apply regularization to the loss
            regularization_penalty = 0.0
            for layer in self.layers:
                regularization_penalty += self.regularization.compute_penalty(layer.W)
            return base_loss + regularization_penalty, accuracy

        return base_loss, accuracy

    def fit(
            self,
            X: NDArray,
            Y: NDArray,
            epochs: int = 50,
            batch_size: int = 32,
            validation_data: Optional[Tuple[NDArray, NDArray]] = None,
            early_stopping: Optional[EarlyStopping] = None,
    ) -> Dict[str, List[Any]]:
        # prepare training and validation data
        X_train, Y_train, X_val, Y_val = self.prepare_data(X, Y, validation_data)
        self.N = X_train.shape[0]
        self.batch_size = batch_size

        history: Dict[str, List[Any]] = {
            "loss": [],
            "accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
        }

        for epoch in range(epochs):
            # Shuffle the datasets
            indices = np.arange(self.N)
            np.random.shuffle(indices)
            X_shuffle = X_train[indices]
            Y_shuffle = Y_train[indices]

            # Iterate over batches
            for start_batch in range(0, self.N, batch_size):
                end_batch = min(start_batch + batch_size, self.N)
                X_batch = X_shuffle[start_batch:end_batch]
                Y_batch = Y_shuffle[start_batch:end_batch]

                # Feedforward & Backpropagation
                # Notice: input & output set into properties of the model in feedforward
                self.feedforward(X_batch)
                self.backpropagation(Y_batch)

                # Update the parameters of the model
                self.update_parameters()

            # Calculate loss and accuracy for training and validation data
            loss, accuracy = self.calculate_loss_accuracy(X_train, Y_train)
            val_loss, val_accuracy = self.calculate_loss_accuracy(X_val, Y_val)

            history["loss"].append(loss)
            history["accuracy"].append(accuracy)
            history["val_loss"].append(val_loss)
            history["val_accuracy"].append(val_accuracy)

            print(
                f"Epoch {epoch + 1}/{epochs} - "
                f"Loss: {loss:.4f}, Accuracy: {accuracy:.4f} - "
                f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}"
            )

            if early_stopping is not None:
                W = [layer.W for layer in self.layers]
                b = [layer.b for layer in self.layers]

                should_stop = early_stopping.on_epoch_end(
                    epoch=epoch,
                    current_value=val_loss,
                    weights=W,
                    bias=b
                )
                if should_stop:
                    print(f"Early stopping at epoch {epoch + 1}")
                    if early_stopping.is_store:
                        for i, layer in enumerate(self.layers):
                            layer.W = early_stopping.best_weights[i]
                            layer.b = early_stopping.best_bias[i]
                    break

        return history

    def load(self, path: str = "./saved") -> None:
        for i, layer in enumerate(self.layers):
            w_path = f"{path}/layer_{i}_weights.npy"
            b_path = f"{path}/layer_{i}_biases.npy"
            if os.path.exists(w_path) and os.path.exists(b_path):
                layer.W = np.load(w_path)
                layer.b = np.load(b_path)
            else:
                print("Path does not exist or files are missing:", w_path, b_path)

    def save(self, path: str = "./saved") -> None:
        for i, layer in enumerate(self.layers):
            np.save(f"{path}/layer_{i}_weights.npy", layer.W)
            np.save(f"{path}/layer_{i}_biases.npy", layer.b)
