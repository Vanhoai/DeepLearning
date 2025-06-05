import numpy as np

mean = np.array([(1, 1), (5, 5)])
cov = np.array([[1, 0], [0, 1]])


def prepare(num: int = 500, groups=2):
    np.random.seed(5)
    X = np.zeros((num * groups, 2))
    y = np.zeros((num * groups))

    for idx, center in enumerate(mean):
        points = np.random.multivariate_normal(mean=center, cov=cov, size=num)
        X[idx * num:(idx + 1) * num, :] = points
        y[idx * num:(idx + 1) * num] = idx

    datas = X.T
    labels = y.reshape(1, -1)

    # shuffle the data
    indices = np.arange(datas.shape[1])
    np.random.shuffle(indices)

    return datas[:, indices], labels[:, indices]


N = 1000
g = 2


def datasets():
    X, y = prepare(num=int(N / g), groups=g)

    assert X.shape == (2, N)
    assert y.shape == (1, N)

    y = y.reshape(-1)

    # One-hot encoding
    Y = np.zeros((N, g))
    for i in range(N):
        Y[i, int(y[i])] = 1

    Y = Y.T
    assert Y.shape == (g, N)
    return X, Y


class NeuralNetwork:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

        # Layer 1
        self.W1 = np.random.randn(2, 2) * np.sqrt(2.0 / 2)
        self.b1 = np.zeros((2, 1))

        # Layer 2
        self.W2 = np.random.randn(2, 2) * np.sqrt(2.0 / 2)
        self.b2 = np.zeros((2, 1))

        self.A0 = None
        self.Z1 = None
        self.A1 = None
        self.Z2 = None
        self.A2 = None

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def relu_derivative(x):
        return np.where(x > 0, 1, 0)

    @staticmethod
    def softmax(Z):
        exp = np.exp(Z - np.max(Z, axis=0, keepdims=True))
        return exp / exp.sum(axis=0, keepdims=True)

    @staticmethod
    def cross_entropy_loss(Y_true, Y_pred):
        # Y:        R(2 X 1000)
        # Y_pred:   R(2 X 1000)
        epsilon = 1e-15
        Y_pred = np.clip(Y_pred, epsilon, 1 - epsilon)
        return - np.mean(np.sum(Y_true * np.log(Y_pred), axis=0))

    def forward(self, X):
        # A0: R(2 X 1000)
        # W1: R(2 X 2)
        # b1: R(2 X 1)
        # Z1: R(2 X 1000) => A1: R(2 X 1000)
        # W2: R(2 X 2)
        # b2: R(2 X 1)
        # Z2: R(2 X 1000) => A2: R(2 X 1000)

        self.A0 = X

        # Layer 1
        self.Z1 = self.W1.T @ self.A0 + self.b1
        self.A1 = self.relu(self.Z1)

        # Layer 2
        self.Z2 = self.W2.T @ self.A1 + self.b2
        self.A2 = self.softmax(self.Z2)

        return self.A2

    def backpropagation(self, Y, A2):
        M = Y.shape[1]
        # Layer 2
        # A2:                               R(2 X 1000)
        # Y:                                R(2 X 1000)
        # dZ2 = A2 - Y                      R(2 X 1000)
        # dW2 = (A1 @ dZ2.T) / N            R(2 X 2)
        # db2 = (np.sum(dZ2, axis=1)) / N   R(2 X 1)

        # Layer 1
        # dA1 = W2 @ dZ2                    R(2 X 1000)
        # dZ1 = dA1 * ReLU_Derivative(Z1)   R(2 X 1000)
        # dW1 = (A0 @ dZ1.T) / N            R(2 X 2)
        # db1 = (np.sum(dZ1, axis=1)) / N   R(2 X 1)

        # Layer 2
        dZ2 = A2 - Y
        dW2 = (self.A1 @ dZ2.T) / M
        db2 = np.sum(dZ2, axis=1, keepdims=True) / M

        # Layer 1
        dA1 = self.W2 @ dZ2
        dZ1 = dA1 * self.relu_derivative(self.Z1)
        dW1 = (self.A0 @ dZ1.T) / M
        db1 = np.sum(dZ1, axis=1, keepdims=True) / M

        # Update weights and biases
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1

    def fit(self, X, Y, epochs=30):
        for epoch in range(0, epochs):
            # feedforward
            A2 = self.forward(X)
            # Compute loss (cross-entropy)
            loss = self.cross_entropy_loss(Y, A2)
            if epoch % 1000 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")

            # Backpropagation
            self.backpropagation(Y, A2)

    def predict(self, X):
        A2 = self.forward(X)
        return np.argmax(A2, axis=0)

# X_train, Y_train = datasets()
# T = 100
# X_test = X_train[:, :T]
# Y_test = Y_train[:, :T]
#
# X_train = X_train[:, T:]
# Y_train = Y_train[:, T:]
#
# assert X_train.shape == (2, N - T)
# assert Y_train.shape == (2, N - T)
# assert X_test.shape == (2, T)
# assert Y_test.shape == (2, T)
#
# nn = NeuralNetwork(learning_rate=0.01)
# nn.fit(X_train, Y_train, epochs=10000)
#
# predictions = nn.predict(X_test)
# accuracy = np.mean(predictions == np.argmax(Y_test, axis=0))
# print(f"Accuracy: {accuracy:.4f}")
