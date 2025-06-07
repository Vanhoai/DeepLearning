import numpy as np
from keras import datasets
import matplotlib.pyplot as plt

from src.nn.layer import ReLu, Softmax
from src.nn.model import Sequential
from src.nn.loss import CrossEntropy
from src.nn.optimizer import SGD
from src.nn.early_stopping import EarlyStopping, MonitorEarlyStopping


def load_fashion_mnist_from_keras():
    (XTrain, YTrain), (XTest, YTest) = datasets.fashion_mnist.load_data()

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


def plot_history(history):
    plt.figure(figsize=(12, 5))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(history["loss"], label="Training Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history["accuracy"], label="Training Accuracy")
    plt.plot(history["val_accuracy"], label="Validation Accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    X_train, Y_train, X_test, Y_test = load_fashion_mnist_from_keras()

    # X = R(N, d)
    # Y = R(N, classes)
    # N = number of samples
    N, d = X_train.shape
    classes = Y_train.shape[1]
    new_training = False

    # Layer dimensions
    d1 = 256
    d2 = 128

    # Hyperparameters
    eta = 1e-1
    momentum = 0.9
    nesterov = True
    weight_decay = 0.0001
    regularization_lamda = 0.01

    layers = [ReLu(d, d1), ReLu(d1, d2), Softmax(d2, classes)]
    model = Sequential(
        layers=layers,
        loss=CrossEntropy(),
        optimizer=SGD(eta, momentum, nesterov, weight_decay),
        regularization=None,
        regularization_lambda=regularization_lamda,
    )

    if not new_training:
        # Load the model from saved state
        print("Loading model from saved state...")
        model.load("./saved")

    early_stopping = EarlyStopping(
        patience=20,
        min_delta=0.1,
        monitor=MonitorEarlyStopping.VAL_ACCURACY,
        is_store=True
    )

    hist = model.fit(
        X_train,
        Y_train,
        epochs=30,
        batch_size=256,
        early_stopping=early_stopping,
    )
    plot_history(hist)
    model.save("./saved")

    # Evaluate the model on the test set
    test_loss, test_accuracy = model.calculate_loss_accuracy(X_test, Y_test)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
