import numpy as np


def relu(x):
    return np.maximum(0, x)


def softmax(x):
    exp = np.exp(x - np.max(x, axis=0, keepdims=True))
    return exp / np.sum(exp, axis=0, keepdims=True)


def relu_derivative(x):
    return np.where(x > 0, 1, 0)


N = 1000
d = 2

np.random.seed(5)
X = np.random.randn(d, N)
A0 = X

# feedforward
W1 = np.random.randn(2, 2)
W2 = np.random.randn(2, 2)
b1 = np.zeros((2, 1))
b2 = np.zeros((2, 1))

Z1 = W1.T @ A0 + b1 # R(2x1000)
A1 = relu(Z1)       # R(2x1000)

Z2 = W2.T @ A1 + b2
A2 = softmax(Z2)

# backpropagation
y = np.random.randint(0, 2, N)
Y = np.eye(2)[y].T

dZ2 = A2 - Y # R(2x1000)

dW2 = A1 @ dZ2.T / N
db2 = np.sum(dZ2, axis=1, keepdims=True) / N

dA1 = W2 @ dZ2
dZ1 = dA1 * relu_derivative(Z1)
dW1 = A0 @ dZ1.T / N
db1 = np.sum(dZ1, axis=1, keepdims=True) / N

# update weights
lr = 0.001

W2 = W2 - lr * dW2
W1 = W1 - lr * dW1
b2 = b2 - lr * db2
b1 = b1 - lr * db1

print(W2)
print(W1)