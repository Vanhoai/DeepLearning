import time
import unittest
import numpy as np
from src.nn.datasets import load_2d_datasets, load_mnist_dataset
from src.nn.layer import ReLu, Softmax
from src.nn.model import Sequential
from src.nn.loss import CrossEntropy
from src.nn.optimizer import Adam
from src.nn.early_stopping import EarlyStopping, MonitorEarlyStopping


class TestNeuralNetworkV3(unittest.TestCase):
    def load_2d_datasets(self, N=None, classes=None, d=None):
        assert N is not None, "N must be specified"
        assert classes is not None, "classes must be specified"
        assert d is not None, "d must be specified"

        mean = np.array([[1, 1], [1, 6], [6, 1], [6, 6]])
        X, Y = load_2d_datasets(N=N, classes=classes, d=d, mean=mean)
        T = N * 0.2

        X_test, Y_test = X[: int(T)], Y[: int(T)]
        X_train, Y_train = X[int(T) :], Y[int(T) :]
        return X_train, Y_train, X_test, Y_test

    def load_mnist_datasets(self):
        X_train, Y_train, X_test, Y_test = load_mnist_dataset()
        return X_train, Y_train, X_test, Y_test

    def build_model_2d_datasets(self, classes=4, d=2):
        d1 = 5

        eta = 1e-2
        layers = [ReLu(d, d1), Softmax(d1, classes)]
        model = Sequential(
            layers=layers,
            loss=CrossEntropy(),
            optimizer=Adam(eta=eta),
            regularization=None,
        )

        model.summary(d)
        return model

    def build_model_mnist(self):
        d = 784  # 28x28 images flattened
        d1 = 512
        d2 = 256
        classes = 10
        eta = 1e-3

        layers = [ReLu(d, d1), ReLu(d1, d2), Softmax(d2, classes)]
        model = Sequential(
            layers=layers,
            loss=CrossEntropy(),
            optimizer=Adam(eta=eta),
            regularization=None,
        )

        model.summary(d)
        return model

    def evaluate_model(self, model, history, X_test, Y_test):
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

        A = model.predict(X_test)
        accuracy = model.evaluate(Y_test, A)
        print("Test accuracy:", accuracy)
        self.assertGreaterEqual(
            accuracy,
            0.9,
            "Test accuracy should be greater than or equal to 0.9",
        )

        print(f"Test accuracy : {accuracy:.4f} >= 0.9 ✅")

    def test_training_2d_datasets(self):
        N = 1000
        classes = 4
        d = 2

        X_train, Y_train, X_test, Y_test = self.load_2d_datasets(
            N=N,
            classes=classes,
            d=d,
        )

        model = self.build_model_2d_datasets(classes=classes, d=d)
        early_stopping = EarlyStopping(
            patience=20,
            min_delta=0.1,
            monitor=MonitorEarlyStopping.VAL_ACCURACY,
            is_store=True,
        )

        start = time.time()
        history = model.fit(
            X_train,
            Y_train,
            epochs=1000,
            verbose=True,
            frequency=100,
            batch_size=256,
            early_stopping=early_stopping,
        )
        end = time.time()
        print(f"Training time: {end - start:.2f} seconds")
        self.evaluate_model(model, history, X_test, Y_test)

    def test_training_mnist(self):
        X_train, Y_train, X_test, Y_test = self.load_mnist_datasets()

        model = self.build_model_mnist()
        early_stopping = EarlyStopping(
            patience=20,
            min_delta=0.1,
            monitor=MonitorEarlyStopping.VAL_ACCURACY,
            is_store=True,
        )

        start = time.time()
        history = model.fit(
            X_train,
            Y_train,
            epochs=20,
            verbose=True,
            frequency=5,
            batch_size=256,
            early_stopping=early_stopping,
        )

        end = time.time()
        print(f"Training time: {end - start:.2f} seconds")
        self.evaluate_model(model, history, X_test, Y_test)


if __name__ == "__main__":
    unittest.main()
