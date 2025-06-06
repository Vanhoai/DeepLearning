import numpy as np
from numpy.typing import NDArray
from abc import ABC, abstractmethod
from src.nn.activation import (
    Activation,
    ReLUActivation,
    SigmoidActivation,
    SoftmaxActivation,
    ELUActivation,
    TanhActivation,
    LeakyReLUActivation,
    LinearActivation,
)
from typing import Optional


class Layer(ABC):
    def __init__(self, di: int, do: int, activation: Activation) -> None:
        self.di = di
        self.do = do

        # Activation function, can be set later
        self.activation: Activation = activation

        # Initialize weights and biases
        self.W = np.random.randn(di, do) * np.sqrt(2.0 / di)
        self.b = np.zeros((1, do))

        self.Z: Optional[NDArray] = None  # Pre-activation output
        self.A: Optional[NDArray] = None  # Post-activation output
        self.dW: Optional[NDArray] = None  # Gradient of weights
        self.db: Optional[NDArray] = None  # Gradient of biases

    def derivativeZ(self) -> NDArray:
        assert self.Z is not None, "Z must be computed before calling derivativeZ"
        return self.activation.derivative(self.Z)

    def forward(self, X: NDArray) -> NDArray:
        self.Z = X @ self.W + self.b
        self.A = self.activation(self.Z)  # type: ignore[return-value]
        assert self.A is not None, "Activation output A must be computed"
        return self.A


class Linear(Layer):
    def __init__(self, di: int, do: int) -> None:
        super().__init__(di, do, LinearActivation())


class ReLu(Layer):
    def __init__(self, di: int, do: int) -> None:
        super().__init__(di, do, ReLUActivation())


class Sigmoid(Layer):
    def __init__(self, di: int, do: int) -> None:
        super().__init__(di, do, SigmoidActivation())


class Softmax(Layer):
    def __init__(self, di: int, do: int) -> None:
        super().__init__(di, do, SoftmaxActivation())


class Tanh(Layer):
    def __init__(self, di: int, do: int) -> None:
        super().__init__(di, do, TanhActivation())


class ELU(Layer):
    def __init__(self, di: int, do: int, alpha: float = 1.0) -> None:
        super().__init__(di, do, ELUActivation(alpha))


class LeakyReLU(Layer):
    def __init__(self, di: int, do: int, alpha: float = 0.01) -> None:
        super().__init__(di, do, LeakyReLUActivation(alpha))
