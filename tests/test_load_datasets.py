import sys
import os

src = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, src)

import unittest
import numpy as np
from nn.datasets import load_2d_datasets, load_mnist_dataset


class TestLoadDatasets(unittest.TestCase):
    def setUp(self) -> None:
        self.N = 1000  # Number of samples
        self.classes = 4  # Number of classes
        self.d = 2  # Dimensions

    def load_2d_datasets(self, N=None, classes=None, d=None):
        mean = np.array([[1, 1], [1, 6], [6, 1], [6, 6]])
        X, Y = load_2d_datasets(N=self.N, classes=self.classes, d=self.d, mean=mean)

        return X, Y

    def test_load_2d_datasets(self):
        X, Y = self.load_2d_datasets()

        self.assertEqual(X.shape, (self.N, self.d))
        self.assertEqual(Y.shape, (self.N, self.classes))

        print("2D Dataset loaded done ✅")

    def test_split_2d_datasets(self):
        X, Y = self.load_2d_datasets()

        T = self.N * 0.2
        X_test, Y_test = X[: int(T)], Y[: int(T)]
        X_train, Y_train = X[int(T) :], Y[int(T) :]

        self.assertEqual(X_test.shape, (int(T), self.d))
        self.assertEqual(Y_test.shape, (int(T), self.classes))
        self.assertEqual(X_train.shape, (self.N - int(T), self.d))
        self.assertEqual(Y_train.shape, (self.N - int(T), self.classes))

        print("2D Dataset split done ✅")

    def test_load_mnist_dataset(self):
        N = 60000
        T = 10000
        d = 784  # 28x28 images flattened
        classes = 10

        X_train, Y_train, X_test, Y_test = load_mnist_dataset()

        self.assertEqual(X_train.shape, (N, d))
        self.assertEqual(Y_train.shape, (N, classes))
        self.assertEqual(X_test.shape, (T, d))
        self.assertEqual(Y_test.shape, (T, classes))

        print("MNIST Dataset loaded done ✅")


if __name__ == "__main__":
    unittest.main()
