import numpy as np

def relu(X):
    return np.maximum(0, X)

def relu_derivative(X):
    return np.where(X > 0, 1, 0)

def softmax(Z):
    """
    Softmax activation function
    Z: R(N X d) = R(1000 X 4)
    Formula:
                Softmax(Z) = e^Zi / (j ∈ d) Σ e^Zj
    where Zi is the i-th element of Z and d is the number of classes.
    This function computes the softmax activation for each row in Z.
    Return: R(N X d) = R(1000 X 4)
    """
    exp = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)

def cross_entropy_loss(Y_true, Y_pred):
    # Y_true:  R(N X d)
    # Y_pred:  R(N X d)
    epsilon = 1e-15
    Y_pred = np.clip(Y_pred, epsilon, 1 - epsilon)
    return - np.mean(np.sum(Y_true * np.log(Y_pred), axis=1))

class NeuralNetwork:
    def __init__(self, learning_rate=1e-3):
        # Learning rate
        self.eta = learning_rate

        # Number units in each layer
        d0 = 2
        d1 = 5
        d2 = 4

        # Initializing weights and biases
        # W1: R(2 X 5)
        # b1: R(1 X 5)
        # W2: R(5 X 4)
        # b2: R(1 X 4)
        self.W1 = np.random.randn(d0, d1) * np.sqrt(2.0 / d0)
        self.b1 = np.zeros((1, d1))

        self.W2 = np.random.randn(d1, d2) * np.sqrt(2.0 / d1)
        self.b2 = np.zeros((1, d2))

        # Cache for backpropagation
        self.A0 = None
        self.Z1 = None
        self.A1 = None
        self.Z2 = None
        self.A2 = None

    def forward(self, X):
        """
        Forward pass through the network
        X: (N, d0) = R(1000, 2) = A0

        Mathematical operations:
        Z1 = A0 @ W1 + b1           : (1000, 2) @ (2, 5) + (1, 5) = (1000, 5)
        A1 = relu(Z1)               : (1000, 5)
        Z2 = A1 @ W2 + b2           : (1000, 5) @ (5, 4) + (1, 4) = (1000, 4)
        A2 = softmax(Z2)            : (1000, 4)
        """

        # Layer 1
        self.A0 = X
        self.Z1 = self.A0 @ self.W1 + self.b1
        self.A1 = relu(self.Z1)

        # Layer 2
        self.Z2 = self.A1 @ self.W2 + self.b2
        self.A2 = softmax(self.Z2)

        return self.A2

    def backpropagation(self, Y):
        """
        Backpropagation with correct mathematical derivation
        Chain rule derivation:
        Loss = CrossEntropy(Y, A2)

        Layer 2:
        dZ2 = ∂L/∂Z2 = A2 - Y                       : (1000, 4)
        dW2 = ∂L/∂W2 = A1.T @ dZ2 / N               : (5, 4)
        db2 = ∂L/∂b2 = mean(dZ2, axis=0)            : (1, 4)

        Layer 1:
        dA1 = ∂L/∂A1 = ∂L/∂Z2.∂Z2/∂A1               : (1000, 5)
            = dZ2 @ W2.T
        dZ1 = ∂L/∂Z1 = ∂L/∂A1 * ReLU'(Z1)           : (1000, 5)
        dW1 = ∂L/∂W1 = A0.T @ dZ1 / N               : (2, 5)
        db1 = ∂L/∂b1 = mean(dZ1, axis=0)            : (1, 5)
        """

        N = Y.shape[0]
        # Layer 2
        dZ2 = self.A2 - Y                           # (1000, 4)
        dW2 = self.A1.T @ dZ2 / N                   # (5 X 1000) @ (1000 X 4) / 1000 = (5, 4)
        db2 = np.mean(dZ2, axis=0, keepdims=True)   # Sum by column and keep dimensions: (1, 4)

        # Layer 1
        dA1 = dZ2 @ self.W2.T                       # (1000, 4) @ (4, 5) = (1000, 5)
        dZ1 = dA1 * relu_derivative(self.Z1)        # (1000, 5) * (1000, 5) = (1000, 5)
        dW1 = self.A0.T @ dZ1 / N                   # (2 X 1000) @ (1000 X 5) / 1000 = (2, 5)
        db1 = np.mean(dZ1, axis=0, keepdims=True)   # Sum by column and keep dimensions: (1, 5)

        # Update weights and biases
        self.W2 -= self.eta * dW2
        self.b2 -= self.eta * db2
        self.W1 -= self.eta * dW1
        self.b1 -= self.eta * db1

    def fit(self, X, Y, epochs=10000, verbose=True):
        for epoch in range(epochs):
            # Forward pass
            A2 = self.forward(X)                    # R(1000, 4)

            # Compute loss
            loss = cross_entropy_loss(Y, A2)        # Scalar value
            if verbose and epoch % 1000 == 0:
                accuracy = self.compute_accuracy(Y, A2)
                print(f"Epoch {epoch + 1:5d}/{epochs}, Loss: {loss:.6f}, Accuracy: {accuracy:.4f}")

            # Backward pass
            self.backpropagation(Y)

    @staticmethod
    def compute_accuracy(Y_true, Y_pred):
        """
        Compute accuracy of predictions
        Y_true: (N, C) - one-hot encoded
        Y_pred: (N, C) - predicted probabilities
        """
        y_true_labels = np.argmax(Y_true, axis=1)
        y_pred_labels = np.argmax(Y_pred, axis=1)
        return np.mean(y_true_labels == y_pred_labels)

    def predict(self, X):
        """
        Make predictions
        X: (N, d)
        Forward pass through the network to get predictions
        """
        A2 = self.forward(X)
        return A2