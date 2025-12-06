import torch

from .module import HSModule


class HSSequential(HSModule):
    def __init__(self, *layers):
        super().__init__()
        self.layers = list(layers)
        for idx, layer in enumerate(layers):
            setattr(self, f"layer_{idx}", layer)

    def forward(self, X: torch.Tensor) -> torch.Tensor:  # type: ignore
        for layer in self.layers:
            X = layer(X)

        return X

    def backward(self, grad_output: torch.Tensor) -> torch.Tensor:  # type: ignore
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output)

        return grad_output
