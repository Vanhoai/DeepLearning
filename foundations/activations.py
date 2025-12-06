import torch

from .module import HSModule


class HSActivation(HSModule): ...


class HSReLU(HSActivation):
    """
    Rectified Linear Unit (ReLU) Activation Function
    Formula:        f(x) = max(0, x)
    Derivative:     dL/dx = dL/dy * (x > 0)
    """

    def forward(self, X: torch.Tensor) -> torch.Tensor:  # type: ignore
        self._cache["X"] = X
        return torch.maximum(torch.zeros_like(X), X)

    def backward(self, grad_output: torch.Tensor) -> torch.Tensor:  # type: ignore
        X = self._cache["X"]
        return grad_output * (X > 0).float()


class HSSigmoid(HSActivation):
    """
    Sigmoid Activation Function
    Formula:        f(x) = 1 / (1 + exp(-x))
    Derivative:     dL/dx = dL/dy * y * (1 - y)
    """

    def forward(self, grad_output: torch.Tensor) -> torch.Tensor:  # type: ignore
        output = self._cache["output"]
        return grad_output * output * (1 - output)

    def backward(self, grad_output: torch.Tensor) -> torch.Tensor:  # type: ignore
        output = self._cache["output"]
        return grad_output * output * (1 - output)


class HSSoftmax(HSActivation):
    """
    Softmax Activation Function
    Formula:        f(x_i) = exp(x_i) / Σ exp(x_j)
    Derivative:     dL/dx = itself when combined with cross-entropy loss
    """

    def __init__(self, dim: int = 1):
        super().__init__()
        self.dim = dim

    def forward(self, X: torch.Tensor) -> torch.Tensor:  # type: ignore
        exp_X = torch.exp(X - torch.max(X, dim=self.dim, keepdim=True).values)
        output = exp_X / torch.sum(exp_X, dim=self.dim, keepdim=True)
        self._cache["output"] = output
        return output

    def backward(self, grad_output: torch.Tensor) -> torch.Tensor:  # type: ignore
        # Simplified: Normally, the gradient of softmax is more complex,
        # but when combined with cross-entropy loss, it simplifies to this.
        return grad_output
