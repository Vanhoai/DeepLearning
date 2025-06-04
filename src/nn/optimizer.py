import numpy as np
from numpy.typing import NDArray
from abc import ABC, abstractmethod


class Optimizer(ABC):
    def __init__(self, lr: float):
        self._lr = lr

    @property
    def lr(self) -> float:
        return self._lr

    @lr.setter
    def lr(self, lr: float):
        self._lr = lr

    @abstractmethod
    def update_weights(self, layer, grad_weights): ...

    @abstractmethod
    def update_bias(self, layer, grad_bias): ...


class SGD(Optimizer):
    def __init__(self, lr: float):
        super().__init__(lr)

    def update_weights(self, layer, grad_weights):
        layer.weights -= self._lr * grad_weights

    def update_bias(self, layer, grad_bias):
        layer.bias -= self._lr * grad_bias
