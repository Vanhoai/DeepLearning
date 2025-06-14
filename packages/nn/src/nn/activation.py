from packages.nn.src.nn.common import Gradient
import numpy as np
from abc import abstractmethod
from numpy.typing import NDArray


class Activation(Gradient):
    @abstractmethod
    def __call__(self, X: NDArray) -> NDArray: ...

    @abstractmethod
    def derivative(self, X: NDArray) -> NDArray: ...


class LinearActivation(Activation):
    def __call__(self, X: NDArray) -> NDArray:
        return X

    def derivative(self, X: NDArray) -> NDArray:
        return np.ones_like(X)


class ReLUActivation(Activation):
    def __call__(self, X: NDArray) -> NDArray:
        return np.maximum(X, 0)

    def derivative(self, X: NDArray) -> NDArray:
        return np.where(X > 0, 1, 0)


class SigmoidActivation(Activation):
    def __call__(self, X: NDArray) -> NDArray:
        return 1 / (1 + np.exp(-X))

    def derivative(self, X: NDArray) -> NDArray:
        return X * (1 - X)


class SoftmaxActivation(Activation):
    def __call__(self, X: NDArray) -> NDArray:
        exp = np.exp(X - np.max(X, axis=1, keepdims=True))
        return exp / exp.sum(axis=1, keepdims=True)

    def derivative(self, X: NDArray) -> NDArray:
        return np.ones_like(X)


class TanhActivation(Activation):
    def __call__(self, X: NDArray) -> NDArray:
        return np.tanh(X)

    def derivative(self, X: NDArray) -> NDArray:
        return 1 - np.tanh(X) ** 2


class LeakyReLUActivation(Activation):
    def __init__(self, alpha: float = 0.01):
        self.alpha = alpha

    def __call__(self, X: NDArray) -> NDArray:
        return np.where(X > 0, X, self.alpha * X)

    def derivative(self, X: NDArray) -> NDArray:
        return np.where(X > 0, 1, self.alpha)


class ELUActivation(Activation):
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def __call__(self, X: NDArray) -> NDArray:
        return np.where(X >= 0, X, self.alpha * (np.exp(X) - 1))

    def derivative(self, X: NDArray) -> NDArray:
        return np.where(X >= 0, 1, self.__call__(X) + self.alpha)
