import torch

from .module import HSModule


class HSLayer(HSModule): ...


class HSLinear(HSLayer):
    """
    Fully Connected Layer: y = xW^T + b

    Forward: y = xW^T + b
    Backward:
        dL/dx = dL/dy @ W
        dL/dW = dL/dy^T @ x
        dL/db = sum(dL/dy, axis=0)
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Xavier initialization
        limit = (6 / (in_features + out_features)) ** 0.5
        self.weight = torch.empty(out_features, in_features).uniform_(-limit, limit)

        if bias:
            self.bias = torch.zeros(out_features)
        else:
            self.bias = None

        # Initialize gradients
        self.weight.grad = torch.zeros_like(self.weight)
        if self.bias is not None:
            self.bias.grad = torch.zeros_like(self.bias)

    def forward(self, X: torch.Tensor) -> torch.Tensor:  # type: ignore
        self._cache["X"] = X

        # y = xW^T + b
        output = X @ self.weight.T
        if self.bias is not None:
            output = output + self.bias

        return output

    def backward(self, grad_output: torch.Tensor) -> torch.Tensor:  # type: ignore
        X = self._cache["X"]

        # dL/dW = (dL/dy)^T @ x
        self.weight.grad += grad_output.T @ X

        # dL/db = sum(dL/dy, axis=0)
        if self.bias is not None:
            self.bias.grad += torch.sum(grad_output, dim=0)  # type: ignore

        # dL/dx = dL/dy @ W
        grad_input = grad_output @ self.weight
        return grad_input


class HSFlatten(HSLayer):
    """
    Flatten Layer: Flattens input tensor except for the batch dimension.
    Forward: Reshapes input to (batch_size, -1)
    Backward: Reshapes gradient to original input shape.
    """

    def forward(self, X: torch.Tensor) -> torch.Tensor:  # type: ignore
        self._cache["X"] = X.shape
        batch_size = X.shape[0]
        return X.view(batch_size, -1)

    def backward(self, grad_output: torch.Tensor) -> torch.Tensor:  # type: ignore
        input_shape = self._cache["X"]
        return grad_output.view(input_shape)


class HSDropout(HSLayer):
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(self, X: torch.Tensor) -> torch.Tensor:  # type: ignore
        if self.training and self.p > 0:
            mask = (torch.rand_like(X) > self.p).float()
            self._cache["mask"] = mask
            output = X * mask / (1 - self.p)
        else:
            output = X

        return output

    def backward(self, grad_output: torch.Tensor) -> torch.Tensor:  # type: ignore
        if self.training and self.p > 0:
            mask = self._cache["mask"]
            return grad_output * mask / (1 - self.p)

        return grad_output


class HSConv2D(HSLayer):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # He initialization cho ReLU
        n = in_channels * kernel_size * kernel_size
        self.weight = (
            torch.randn(out_channels, in_channels, kernel_size, kernel_size)
            * (2.0 / n) ** 0.5
        )
        self.bias = torch.zeros(out_channels)

        self.weight.grad = torch.zeros_like(self.weight)
        self.bias.grad = torch.zeros_like(self.bias)

    def forward(self, X: torch.Tensor) -> torch.Tensor:  # type: ignore
        self._cache["X"] = X

        # Dùng torch functional cho đơn giản
        import torch.nn.functional as F

        output = F.conv2d(X, self.weight, self.bias, self.stride, self.padding)
        return output

    def backward(self, grad_output: torch.Tensor) -> torch.Tensor:  # type: ignore
        import torch.nn.functional as F

        X = self._cache["X"]

        # dL/dW: convolve input with grad_output
        self.weight.grad += F.conv2d(
            X.transpose(0, 1),
            grad_output.transpose(0, 1),
            stride=self.stride,
            padding=self.padding,
        ).transpose(0, 1)

        # dL/db
        self.bias.grad += grad_output.sum(dim=(0, 2, 3))  # type: ignore

        # dL/dx: full convolution (transpose)
        grad_input = F.conv_transpose2d(
            grad_output,
            self.weight,
            stride=self.stride,
            padding=self.padding,
        )

        return grad_input


class HSMaxPool2d(HSLayer):
    def __init__(self, kernel_size: int, stride: int | None = None):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size

    def forward(self, X: torch.Tensor) -> torch.Tensor:  # type: ignore
        import torch.nn.functional as F

        # Max pooling with return_indices
        output, indices = F.max_pool2d(
            X, self.kernel_size, self.stride, return_indices=True
        )
        self._cache["indices"] = indices
        self._cache["input_shape"] = X.shape
        return output

    def backward(self, grad_output: torch.Tensor) -> torch.Tensor:  # type: ignore
        import torch.nn.functional as F

        indices = self._cache["indices"]
        input_shape = self._cache["input_shape"]

        grad_input = F.max_unpool2d(
            grad_output,
            indices,
            self.kernel_size,  # type: ignore
            self.stride,  # type: ignore
            output_size=input_shape,
        )

        return grad_input
