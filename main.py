import numpy as np
from keras import datasets

from src.nn.layer import ReLu, Softmax
from src.nn.model import Sequential
from src.nn.loss import CrossEntropy
from src.nn.optimizer import SGD


def load_mnist_from_keras():
    (XTrain, YTrain), (XTest, YTest) = datasets.mnist.load_data()
    # XTrain: (60000, 28, 28)
    # YTrain: (60000,)

    # Reshape and normalize the data
    XTrain = XTrain.reshape(XTrain.shape[0], -1) / 255.0  # Normalize and flatten
    XTest = XTest.reshape(XTest.shape[0], -1) / 255.0  # Normalize and flatten

    # Convert labels to one-hot encoding
    YTrain = np.eye(10)[YTrain]  # One-hot encoding for training labels
    YTest = np.eye(10)[YTest]  # One-hot encoding for test labels

    assert XTrain.shape == (60000, 784)
    assert YTrain.shape == (60000, 10)
    assert XTest.shape == (10000, 784)
    assert YTest.shape == (10000, 10)

    return XTrain, YTrain, XTest, YTest


if __name__ == "__main__":
    X_train, Y_train, X_test, Y_test = load_mnist_from_keras()

    d = 784  # Input dimension (28x28 images flattened)
    classes = 10  # Number of classes (digits 0-9)
    eta = 1e-3  # Learning rate
    momentum = 0.9  # Momentum for SGD
    nesterov = True  # Use Nesterov momentum
    weight_decay = 0.0001  # Weight decay for regularization
    regularization = 0.0001  # Regularization strength

    print(f"X_train shape: {X_train.shape}")
    print(f"Y_train shape: {Y_train.shape}")

    layers = [ReLu(d, 256), ReLu(256, 128), Softmax(128, classes)]
    model = Sequential(
        layers=layers,
        loss=CrossEntropy(),
        optimizer=SGD(eta, momentum, nesterov, weight_decay),
        regularization=regularization,
    )

    model.fit(X_train, Y_train, epochs=20, batch_size=256)
