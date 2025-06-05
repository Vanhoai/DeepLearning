from abc import ABC, abstractmethod
from numpy.typing import NDArray
import numpy as np
from src.nn.optimizer import Optimizer


class Layer(ABC):
    @property
    @abstractmethod
    def output(self): ...

    @property
    @abstractmethod
    def weights(self): ...

    @weights.setter
    def weights(self, weights: NDArray): ...

    @abstractmethod
    def forward(self, input_tensor: NDArray) -> NDArray: ...

    @abstractmethod
    def build(self, input_tensor: NDArray): ...

    @abstractmethod
    def update(self, optimizer: Optimizer): ...


class Dense(Layer):
    def __init__(self, units: int):
        self._units = units
        self._input_units = None
        self._weights = None
        self._bias = None
        self._output = None
        self._dw = None
        self._db = None

    def __repr__(self):
        return f"Dense(units={self._units}) with weights shape {self._weights.shape if self._weights is not None else "None"} and bias shape {self._bias.shape if self._bias is not None else "None"}"

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, weights: NDArray):
        self._weights = weights

    @property
    def bias(self):
        return self._bias

    @bias.setter
    def bias(self, bias: NDArray):
        self._bias = bias

    @property
    def dw(self):
        return self._dw

    @dw.setter
    def dw(self, gradients: NDArray):
        self._dw = gradients

    @property
    def db(self):
        return self._db

    @db.setter
    def db(self, gradients: NDArray):
        self._db = gradients

    @property
    def output(self):
        return self._output

    def build(self, input_tensor: NDArray):
        self._input_units = input_tensor.shape[0]
        normalized = np.sqrt(2.0 / self._input_units)
        self._weights = np.random.randn(self._input_units, self._units) * normalized
        self._bias = np.zeros((self._units, 1))

    # Weights:      R(l - 1 X l)
    # Bias:         R(l X 1)
    def forward(self, input_tensor: NDArray) -> NDArray:
        if self._weights is None:
            self.build(input_tensor)

        self._output = self._weights.T @ input_tensor + self._bias
        return self._output

    def update(self, optimizer: Optimizer):
        optimizer.update_weights(self, self.dw)
        optimizer.update_bias(self, self.db)
