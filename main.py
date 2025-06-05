import numpy as np
import matplotlib.pyplot as plt
from datasets import datasets
from neural_network import NeuralNetwork

N = 10000  # Total number of samples
classes = 4  # Number of classes
d = 2  # Dimensions

def train_test_split():
    # Load datasets
    X_data, Y_data = datasets(N, classes, d)

    # Train test split
    T = int(0.2 * N)
    XTest, YTest = X_data[:T], Y_data[:T]
    XTrain, YTrain = X_data[T:], Y_data[T:]

    assert XTest.shape == (T, d)
    assert YTest.shape == (T, classes)

    assert XTrain.shape == (N - T, d)
    assert YTrain.shape == (N - T, classes)

    return XTrain, YTrain, XTest, YTest

def plot_data(XTrain, YTrain, XTest, YTest):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].scatter(XTrain[:, 0], XTrain[:, 1], c=np.argmax(YTrain, axis=1), cmap='viridis', s=10)
    axs[0].set_title('Training Data')
    axs[0].set_xlabel('X')
    axs[0].set_ylabel('Y')
    axs[1].scatter(XTest[:, 0], XTest[:, 1], c=np.argmax(YTest, axis=1), cmap='viridis', s=10)
    axs[1].set_title('Test Data')
    axs[1].set_xlabel('X')
    axs[1].set_ylabel('Y')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    X_train, Y_train, X_test, Y_test = train_test_split()
    nn = NeuralNetwork(learning_rate=1e-3)
    nn.fit(X_train, Y_train, epochs=10000)

    # Evaluate the model
    predictions = nn.predict(X_test)  # One-hot encoded predictions
    accuracy = nn.compute_accuracy(Y_test, predictions)
    print(f"Test Accuracy: {accuracy:.4f}")



