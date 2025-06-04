import numpy as np
from abc import abstractmethod
from numpy.typing import NDArray
from src.nn.common import Grad


class Activation(Grad):
    @abstractmethod
    def active(self, input_tensor: NDArray) -> NDArray: ...


class ReLU(Activation):
    def __repr__(self):
        return "ReLU()"

    def active(self, input_tensor: NDArray) -> NDArray:
        return np.maximum(input_tensor, 0)

    def gradient(self, input_tensor: NDArray) -> NDArray:
        output = input_tensor.copy()
        output[input_tensor > 0] = 1
        output[input_tensor <= 0] = 0
        return output


class Sigmoid(Activation):
    def __repr__(self):
        return "Sigmoid()"

    def active(self, input_tensor: NDArray) -> NDArray:
        return 1.0 / (1 + np.exp(-input_tensor))

    def gradient(self, input_tensor: NDArray) -> NDArray:
        sig = self.active(input_tensor)
        return sig * (1 - sig)
