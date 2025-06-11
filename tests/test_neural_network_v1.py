import time
import unittest
import numpy as np
import matplotlib.pyplot as plt
from src.nn.pure.neural_network_v1 import NeuralNetworkV1
from src.nn.datasets import load_2d_datasets


class TestNeuralNetworkV1(unittest.TestCase):
    def setUp(self) -> None:
        self.N = 1000
        self.classes = 4
        self.d = 2

    def load_2d_datasets(self):
        mean = np.array([[1, 1], [1, 6], [6, 1], [6, 6]])
        X, Y = load_2d_datasets(N=self.N, classes=self.classes, d=self.d, mean=mean)
        T = self.N * 0.2  # type: ignore

        X_test, Y_test = X[: int(T)], Y[: int(T)]
        X_train, Y_train = X[int(T) :], Y[int(T) :]
        return X_train, Y_train, X_test, Y_test

    def build_model_2d_datasets(self):
        eta = 1e-2
        nn = NeuralNetworkV1(learning_rate=eta)
        return nn

    def test_fit_2d_datasets(self):
        X_train, Y_train, X_test, Y_test = self.load_2d_datasets()
        nn = self.build_model_2d_datasets()

        start = time.time()
        history = nn.fit(
            X_train,
            Y_train,
            epochs=10000,
            verbose=True,
            frequency=1000,
        )
        end = time.time()
        print(f"Training time for 2d dataset: {end - start:.2f} seconds")

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


if __name__ == "__main__":
    unittest.main()
