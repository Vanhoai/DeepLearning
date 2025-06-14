import numpy as np
from keras import datasets
from matplotlib import pyplot as plt
from typing import Optional
from numpy.typing import NDArray

cov = np.array([[1, 0], [0, 1]])


def prepare(num, groups, mean: Optional[NDArray] = None):
    assert mean is not None, "Mean must be provided for dataset generation"
    assert (
        groups == mean.shape[0]
    ), "Number of groups must match the number of means provided"

    X = np.zeros((num * groups, 2))
    y = np.zeros((num * groups))

    for idx, center in enumerate(mean):
        points = np.random.multivariate_normal(mean=center, cov=cov, size=num)
        X[idx * num : (idx + 1) * num, :] = points
        y[idx * num : (idx + 1) * num] = idx

    data = X.T
    labels = y.reshape(1, -1)

    # shuffle the data
    indices = np.arange(data.shape[1])
    np.random.shuffle(indices)

    return data[:, indices], labels[:, indices]


def load_2d_datasets(N, classes, d, mean: NDArray):
    X, y = prepare(num=int(N / classes), groups=classes, mean=mean)

    X = X.T
    y = y.reshape(-1)

    # One-hot encoding
    Y = np.zeros((N, classes))
    for i in range(N):
        Y[i, int(y[i])] = 1

    assert X.shape == (N, d)  # (N, dimensions)
    assert Y.shape == (N, classes)  # (N, classes)

    return X, Y


def load_mnist_dataset():
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


# def run_own_neural_network():
#     X_train, Y_train, X_test, Y_test = load_mnist_from_keras()

#     N = 60000
#     classes = 10  # Number of classes for MNIST
#     d = 784  # Dimensions for MNIST (28x28 flattened)

#     layers = [(d, ReLU()), (256, ReLU()), (128, ReLU()), (classes, Softmax())]
#     nn = NeuralNetworkV2(
#         layers=layers,
#         loss=CrossEntropyLoss(),
#         eta=1e-2,
#     )
#     nn.fit(X_train, Y_train, epochs=20, batch_size=256)
#     A = nn.predict(X_test)
#     accuracy = nn.compute_accuracy(Y_test, A)
#     print(f"accuracy: {accuracy:.4f}")
