import sys
import os
import time

src = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, src)

import unittest
import numpy as np
import matplotlib.pyplot as plt
from nn.pure.neural_network_v2 import NeuralNetworkV2, ReLU, Softmax, CrossEntropyLoss
from nn.datasets import load_2d_datasets, load_mnist_dataset


class TestNeuralNetworkV2(unittest.TestCase):
    def setUp(self) -> None:
        self.N = None  # Number of samples
        self.classes = None  # Number of classes
        self.d = None  # Dimensions

    def setUp2D(self):
        self.N = 1000
        self.classes = 4
        self.d = 2

    def setUpMNIST(self):
        self.N = 60000
        self.classes = 10
        self.d = 784  # 28x28 images flattened

    def load_2d_datasets(self, N=None, classes=None, d=None):
        self.setUp2D()

        mean = np.array([[1, 1], [1, 6], [6, 1], [6, 6]])
        X, Y = load_2d_datasets(N=self.N, classes=self.classes, d=self.d, mean=mean)
        T = self.N * 0.2  # type: ignore

        X_test, Y_test = X[: int(T)], Y[: int(T)]
        X_train, Y_train = X[int(T) :], Y[int(T) :]
        return X_train, Y_train, X_test, Y_test

    def load_mnist_dataset(self):
        X_train, Y_train, X_test, Y_test = load_mnist_dataset()
        return X_train, Y_train, X_test, Y_test

    def build_model_2d_datasets(self):
        d = self.d
        d1 = 5
        classes = self.classes

        layers = [(d, ReLU()), (d1, ReLU()), (classes, Softmax())]
        eta = 1e-2
        nn = NeuralNetworkV2(
            layers=layers,
            loss=CrossEntropyLoss(),
            eta=eta,
        )
        return nn

    def build_model_mnist_dataset(self):
        d = self.d
        d1 = 512
        d2 = 256
        classes = self.classes

        layers = [(d, ReLU()), (d1, ReLU()), (d2, ReLU()), (classes, Softmax())]
        eta = 1e-2
        nn = NeuralNetworkV2(
            layers=layers,
            loss=CrossEntropyLoss(),
            eta=eta,
        )

        return nn

    def test_fit_2d_datasets(self):
        self.setUp2D()

        X_train, Y_train, X_test, Y_test = self.load_2d_datasets()
        nn = self.build_model_2d_datasets()

        start = time.time()
        history = nn.fit(
            X_train,
            Y_train,
            epochs=1000,
            batch_size=32,
            verbose=True,
            frequency=100,
        )
        end = time.time()
        print(f"Training time for 2d datasets: {end - start:.2f} seconds")

        # ensure history contains loss and accuracy
        self.assertIn("loss", history)
        self.assertIn("accuracy", history)
        print("Training successful ✅")

        # check accuracy more than 0.9
        accuracy = max(history["accuracy"])
        self.assertGreaterEqual(
            accuracy,
            0.9,
            "Accuracy should be greater than or equal to 0.9",
        )

        # check loss less than 0.1
        loss = min(history["loss"])
        self.assertLessEqual(
            loss,
            0.1,
            "Loss should be less than or equal to 0.1",
        )

        print(f"Training accuracy : {accuracy:.4f} >= 0.9 ✅")
        print(f"Training loss : {loss:.4f} <= 0.1 ✅")

        A = nn.predict(X_test)
        accuracy = nn.compute_accuracy(Y_test, A)
        print("Test accuracy:", accuracy)
        self.assertGreaterEqual(
            accuracy,
            0.9,
            "Test accuracy should be greater than or equal to 0.9",
        )

        print(f"Test accuracy : {accuracy:.4f} >= 0.9 ✅")

    def test_fit_mnist_dataset(self):
        self.setUpMNIST()

        X_train, Y_train, X_test, Y_test = self.load_mnist_dataset()
        nn = self.build_model_mnist_dataset()

        start = time.time()
        history = nn.fit(
            X_train,
            Y_train,
            epochs=100,
            batch_size=512,
            verbose=True,
            frequency=10,
        )
        end = time.time()
        print(f"Training time for MNIST dataset: {end - start:.2f} seconds")

        # ensure history contains loss and accuracy
        self.assertIn("loss", history)
        self.assertIn("accuracy", history)
        print("Training successful ✅")

        # check accuracy more than 0.9
        accuracy = max(history["accuracy"])
        self.assertGreaterEqual(
            accuracy,
            0.9,
            "Accuracy should be greater than or equal to 0.9",
        )
        print(f"Training accuracy : {accuracy:.4f} >= 0.9 ✅")
        # check loss less than 0.1
        loss = min(history["loss"])
        self.assertLessEqual(
            loss,
            0.1,
            "Loss should be less than or equal to 0.1",
        )
        print(f"Training loss : {loss:.4f} <= 0.1 ✅")
        A = nn.predict(X_test)
        accuracy = nn.compute_accuracy(Y_test, A)
        print("Test accuracy:", accuracy)
        self.assertGreaterEqual(
            accuracy,
            0.9,
            "Test accuracy should be greater than or equal to 0.9",
        )
        print(f"Test accuracy : {accuracy:.4f} >= 0.9 ✅")


if __name__ == "__main__":
    unittest.main()
