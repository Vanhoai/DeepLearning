import numpy as np
from abc import ABC, abstractmethod
from numpy.typing import NDArray
from typing import List, Tuple

from src.nn.layer import Layer
from src.nn.activation import Activation
from src.nn.loss import Loss
from src.nn.optimizer import Optimizer


class Model(ABC):
    @abstractmethod
    def forward(self, input_tensor: NDArray) -> NDArray: ...

    @abstractmethod
    def backward(self, y: NDArray): ...

    @abstractmethod
    def update(self): ...

    @abstractmethod
    def fit(
            self,
            X: NDArray,
            y: NDArray,
            epochs: int,
            verbose: bool = False
    ) -> NDArray: ...


class NeuralNetwork(Model):
    def __init__(
            self,
            layers: List[Tuple[Layer, Activation]],
            loss: Loss,
            optimizer=Optimizer
    ):
        self._layers = layers
        self._num_layers = len(layers)
        self._loss = loss
        self._optimizer = optimizer
        self._input = None
        self._output = None

    def __repr__(self):
        for i, (layer, activation) in enumerate(self._layers):
            print(f"Layer {i + 1}: {layer} with activation {activation}")

        return f"NeuralNetwork(layers={self._num_layers}, loss={self._loss})"

    def forward(self, input_tensor: NDArray) -> NDArray:
        output = input_tensor
        for layer, activation in self._layers:
            z = layer.forward(output)
            output = activation.active(z)

        self._output = output
        return output

    def backward(self, y: NDArray):
        da = self._loss.gradient(self._output, y)

        for index in reversed(range(self._num_layers)):
            layer, activation = self._layers[index]

            # da: dJ / da = f"(loss)
            # dz: dJ / dz = dJ / da * da / dz = f"(loss) * f"(activation)
            # dw: dJ / dw = dJ / dz * dz / dw = f"(loss) * f"(activation) * a(l - 1)
            dz = np.multiply(da, activation.gradient(layer.output))

            a_prev = None
            if index == 0:  # First layer
                a_prev = self._input
            else:
                prev_layer, prev_activation = self._layers[index - 1]
                a_prev = prev_activation.active(prev_layer.output)

            layer.dw = (a_prev @ dz.T) / self._num_layers
            layer.db = np.mean(dz, axis=1, keepdims=True)
            da = layer.weights @ dz

    def fit(
            self,
            X: NDArray,
            y: NDArray,
            epochs: int,
            verbose: bool = False
    ):
        for epoch in range(epochs):
            self._input = X
            self.forward(X)
            loss = self._loss.calculate(self._output, y)

            self.backward(y)
            self.update()

            print(f"Epoch: {epoch + 1} / {epochs}, Loss {loss}")

    def update(self):
        for nl in range(0, len(self._layers)):
            self._optimizer.nl = nl
            self._layers[nl][0].update(self._optimizer)
