from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple
from numpy.typing import NDArray
from src.nn.layer import Layer


class Optimizer(ABC):
    def __init__(self, eta: float = 1e-3) -> None:
        self.eta = eta

    @abstractmethod
    def update(self, layer: Layer, dW: NDArray, db: NDArray) -> None:
        layer.W -= self.eta * dW
        layer.b -= self.eta * db


class SGD(Optimizer):
    def __init__(
        self,
        eta: float = 1e-3,
        momentum: float = 0.9,
        nesterov: bool = True,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(eta)

        # Momentum parameters
        self.momentum = momentum
        self.nesterov = nesterov
        self.weight_decay = weight_decay

    # FIXME: Implement Nesterov momentum later
    def update(self, layer: Layer, dW: NDArray, db: NDArray) -> None:
        super().update(layer, dW, db)
