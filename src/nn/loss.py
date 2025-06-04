from abc import abstractmethod
from typing import Any

import numpy as np
from numpy.typing import NDArray
from src.nn.common import Grad

class Loss(Grad):
    @abstractmethod
    def calculate(self, pred: NDArray, y: NDArray) -> NDArray: ...

class BinaryCrossEntropy(Loss):
    def calculate(self, pred: NDArray, y: NDArray) -> NDArray:
        epsilon = 1e-7
        loss = np.multiply(y, np.log(pred + epsilon)) + np.multiply(1 - y, np.log(1 - pred + epsilon))
        return -1 * np.mean(loss)

    def gradient(self, pred: NDArray, y: NDArray) -> NDArray:
        epsilon = 1e-7
        grad = -1 * np.divide(y, pred + epsilon) - np.divide(1 - y, 1 - pred + epsilon)
        return grad