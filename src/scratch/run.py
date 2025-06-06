import numpy as np
import matplotlib.pyplot as plt
from src.scratch.modern_neural_network import (
    ModernNeuralNetwork,
    ReLU,
    Softmax,
    CrossEntropyLoss,
)
from keras import datasets

N = 10000  # Total number of samples
classes = 4  # Number of classes
d = 2  # Dimensions


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


def plot_mnist_data(XTrain, YTrain, XTest, YTest):
    # Plotting 10 random samples from the training and test sets
    fig, axs = plt.subplots(2, 10, figsize=(15, 4))
    for i in range(10):
        axs[0, i].imshow(XTrain[i].reshape(28, 28), cmap="gray")
        axs[0, i].set_title(f"Train: {np.argmax(YTrain[i])}")
        axs[0, i].axis("off")

        axs[1, i].imshow(XTest[i].reshape(28, 28), cmap="gray")
        axs[1, i].set_title(f"Test: {np.argmax(YTest[i])}")
        axs[1, i].axis("off")

    plt.tight_layout()
    plt.show()


def plot_data(XTrain, YTrain, XTest, YTest):
    _, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].scatter(
        XTrain[:, 0], XTrain[:, 1], c=np.argmax(YTrain, axis=1), cmap="viridis", s=10
    )
    axs[0].set_title("Training Data")
    axs[0].set_xlabel("X")
    axs[0].set_ylabel("Y")
    axs[1].scatter(
        XTest[:, 0], XTest[:, 1], c=np.argmax(YTest, axis=1), cmap="viridis", s=10
    )
    axs[1].set_title("Test Data")
    axs[1].set_xlabel("X")
    axs[1].set_ylabel("Y")

    plt.tight_layout()
    plt.show()


def run_own_neural_network():
    X_train, Y_train, X_test, Y_test = load_mnist_from_keras()

    N = 60000
    classes = 10  # Number of classes for MNIST
    d = 784  # Dimensions for MNIST (28x28 flattened)

    layers = [(d, ReLU()), (256, ReLU()), (128, ReLU()), (classes, Softmax())]
    nn = ModernNeuralNetwork(
        layers=layers,
        loss=CrossEntropyLoss(),
        eta=1e-2,
    )
    nn.fit(X_train, Y_train, epochs=20, batch_size=256)
    A = nn.predict(X_test)
    accuracy = nn.compute_accuracy(Y_test, A)
    print(f"accuracy: {accuracy:.4f}")
