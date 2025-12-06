from abc import ABC, abstractmethod
from typing import Iterator

import torch


class HSOptimizer(ABC):
    def __init__(self, parameters: Iterator[torch.Tensor]):
        self.parameters = list(parameters)

    @abstractmethod
    def step(self): ...

    def zero_grad(self):
        for param in self.parameters:
            if param.grad is not None:
                param.grad.zero_()


class HSSGD(HSOptimizer):
    def __init__(
        self,
        parameters: Iterator[torch.Tensor],
        learning_rate=0.01,
        momentum=0.0,
    ):
        super().__init__(parameters)
        self.learning_rate = learning_rate
        self.momentum = momentum

        self.velocities = [torch.zeros_like(p) for p in self.parameters]

    def step(self):
        for i, param in enumerate(self.parameters):
            if param.grad is None:
                continue

            if self.momentum > 0:
                self.velocities[i] = self.momentum * self.velocities[i] + param.grad
                param.data -= self.learning_rate * self.velocities[i]
            else:
                param.data -= self.learning_rate * param.grad


class HSAdam(HSOptimizer):
    def __init__(
        self,
        parameters: Iterator[torch.Tensor],
        learning_rate=0.001,
        betas=(0.9, 0.999),
        eps=1e-8,
    ):
        super().__init__(parameters)
        self.learning_rate = learning_rate
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.t = 0

        self.m = [torch.zeros_like(p) for p in self.parameters]
        self.v = [torch.zeros_like(p) for p in self.parameters]

    def step(self):
        self.t += 1

        for i, param in enumerate(self.parameters):
            if param.grad is None:
                continue

            # Update moments
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * param.grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * param.grad**2

            # Bias correction
            m_hat = self.m[i] / (1 - self.beta1**self.t)
            v_hat = self.v[i] / (1 - self.beta2**self.t)

            # Update
            param.data -= self.learning_rate * m_hat / (torch.sqrt(v_hat) + self.eps)


# class Optimizer(ABC):
#     def __init__(self, eta: float = 1e-3) -> None:
#         self.eta = eta

#     @abstractmethod
#     def update(self, layer: "Layer", dW: NDArray, db: NDArray) -> None:
#         layer.W -= self.eta * dW
#         layer.b -= self.eta * db


# class SGD(Optimizer):
#     """
#     Stochastic Gradient Descent (SGD)
#     This optimizer updates the parameters using the gradient of the loss function.
#     Formula:
#     w = w - η * g
#     where:
#     - w is the parameter (weights or biases)
#     - η is the learning rate
#     - g is the gradient of the loss function with respect to the parameter

#     Parameters:
#     - eta: Learning rate (default: 1e-3)
#     - momentum: Momentum factor (default: 0.9)
#     - nesterov: Whether to use Nesterov Accelerated Gradient (default: True)
#     - weight_decay: L2 regularization factor (default: 0.0)

#     if use momentum formula becomes:
#     w = w + momentum * v - η * g
#     where:
#     - v is the velocity (previous update)
#     - momentum is the momentum factor
#     - nesterov: If True, uses Nesterov Accelerated Gradient (NAG) which looks ahead at the next position
#     - weight_decay: If non-zero, applies L2 regularization to the weights
#     """

#     def __init__(
#         self,
#         eta: float = 1e-3,
#         momentum: float = 0.9,
#         nesterov: bool = True,
#         weight_decay: float = 0.0,
#     ) -> None:
#         super().__init__(eta)

#         # Momentum parameters
#         self.momentum = momentum
#         self.nesterov = nesterov
#         self.weight_decay = weight_decay

#         # Velocity buffers for momentum
#         self.vW: dict[int, NDArray] = {}
#         self.vb: dict[int, NDArray] = {}

#     def update(self, layer: "Layer", dW: NDArray, db: NDArray) -> None:
#         layer_id = id(layer)
#         if layer_id not in self.vW:
#             self.vW[layer_id] = np.zeros_like(dW)
#             self.vb[layer_id] = np.zeros_like(db)

#         # Apply weight decay (L2 regularization)
#         if self.weight_decay != 0.0:
#             dW += self.weight_decay * layer.W

#         # Update velocity
#         vW = self.vW[layer_id]
#         vb = self.vb[layer_id]

#         # Nesterov Accelerated Gradient (NAG):
#         # velocity = momentum * velocity - learning_rate * g
#         # w = w + momentum * velocity - learning_rate * g
#         if self.nesterov:
#             vW[:] = self.momentum * vW - self.eta * dW
#             vb[:] = self.momentum * vb - self.eta * db

#             layer.W += self.momentum * vW - self.eta * dW
#             layer.b += self.momentum * vb - self.eta * db
#         else:
#             # Gradient Descent with Momentum:
#             # velocity = momentum * velocity - learning_rate * g
#             # w = w + velocity
#             vW[:] = self.momentum * vW - self.eta * dW
#             vb[:] = self.momentum * vb - self.eta * db

#             layer.W += vW
#             layer.b += vb


# class AdaGrad(Optimizer):
#     """
#     Adaptive Gradient Algorithm (AdaGrad)
#     This optimizer adapts the learning rate for each parameter based on the historical gradients.
#     It is particularly useful for dealing with sparse data and features.
#     Formula:
#     G(t) = G(t - 1) + g(t)^2
#     w(t + 1) = w(t) - η / (sqrt(G(t)) + ε)  * g(t)
#     where:
#     - G(t) is the accumulated squared gradients up to the time step t
#     - g(t) is the gradient at time step t
#     - η is the learning rate
#     - ε is a small constant to avoid division by zero

#     Benefits:
#     - Adapts learning rates for each parameter individually
#     - Effective for sparse data and features
#     - Reduces the need for manual learning rate tuning
#     Drawbacks:
#     - Accumulation of squared gradients can lead to overly small learning rates
#     """

#     def __init__(self, eta: float = 1e-3) -> None:
#         super().__init__(eta)

#         # initialize learning rate accumulator
#         self.epsilon = 1e-15

#         self.vtW: dict[int, NDArray] = {}
#         self.vtb: dict[int, NDArray] = {}

#     def update(self, layer: "Layer", dW: NDArray, db: NDArray) -> None:
#         layer_id = id(layer)

#         # Initialize if first time
#         if layer_id not in self.vtW:
#             self.vtW[layer_id] = np.zeros_like(dW)
#             self.vtb[layer_id] = np.zeros_like(db)

#         vtW = self.vtW[layer_id]
#         vtb = self.vtb[layer_id]

#         # Accumulate squared gradients
#         vtW += dW**2
#         vtb += db**2

#         # Apply parameter update with element-wise division
#         layer.W -= self.eta * dW / (np.sqrt(vtW) + self.epsilon)
#         layer.b -= self.eta * db / (np.sqrt(vtb) + self.epsilon)


# class RMSProp(Optimizer):
#     """
#     Root Mean Square Propagation (RMSProp)
#     This optimizer is designed to adapt the learning rate for each parameter based on the average of recent gradients.
#     It helps to stabilize the learning process and is particularly effective for non-stationary objectives.
#     Formula:
#     E[g^2](t) = β * E[g^2](t - 1) + (1 - β) * g(t)^2
#     w(t + 1) = w(t) - η / (sqrt(E[g^2](t)) + ε) * g(t)
#     where:
#     - E[g^2](t) is the exponentially weighted moving average of squared gradients
#     - g(t) is the gradient at time step t
#     - β is the decay rate (typically around 0.9 or 0.95)
#     - g is the gradient
#     - η is the learning rate
#     - ε is a small constant to avoid division by zero
#     """

#     def __init__(self, eta: float = 1e-3, beta: float = 0.9) -> None:
#         super().__init__(eta)

#         # initialize learning rate accumulator
#         self.epsilon = 1e-15
#         self.beta = beta

#         self.vtW: dict[int, NDArray] = {}
#         self.vtb: dict[int, NDArray] = {}

#     def update(self, layer: "Layer", dW: NDArray, db: NDArray) -> None:
#         layer_id = id(layer)

#         # Initialize if first time
#         if layer_id not in self.vtW:
#             self.vtW[layer_id] = np.zeros_like(dW)
#             self.vtb[layer_id] = np.zeros_like(db)

#         vtW = self.vtW[layer_id]
#         vtb = self.vtb[layer_id]

#         # Update moving average of squared gradients
#         vtW[:] = self.beta * vtW + (1 - self.beta) * dW**2
#         vtb[:] = self.beta * vtb + (1 - self.beta) * db**2

#         # Apply parameter update with element-wise division
#         layer.W -= self.eta * dW / (np.sqrt(vtW) + self.epsilon)
#         layer.b -= self.eta * db / (np.sqrt(vtb) + self.epsilon)


# class Adam(Optimizer):
#     """
#     Adam (Adaptive Moment Estimation)
#     This optimizer combines the benefits of momentum and RMSProp.
#     It maintains both the first moment (mean) and the second moment (variance) of the gradients.
#     It is widely used due to its efficiency and effectiveness in training deep neural networks.
#     Formula:
#     m = β1 * m + (1 - β1) * g
#     v = β2 * v + (1 - β2) * g^2
#     m_hat = m / (1 - β1^t)
#     v_hat = v / (1 - β2^t)
#     w = w - η * m_hat / (sqrt(v_hat) + ε)
#     where:
#     - m is the first moment (mean of exponential moving average of gradients)
#     - v is the second moment (mean of exponential moving average of squared gradients)
#     - β1 is the decay rate for the first moment (typically around 0.9)
#     - β2 is the decay rate for the second moment (typically around 0.999)
#     - m_hat and v_hat are bias-corrected estimates of the first and second moments
#     - g is the gradient
#     - η is the learning rate
#     - ε is a small constant to avoid division by zero
#     """

#     def __init__(
#         self,
#         eta: float = 1e-3,
#         beta1: float = 0.9,
#         beta2: float = 0.999,
#         epsilon: float = 1e-8,
#     ):
#         super().__init__(eta)
#         self.beta1 = beta1
#         self.beta2 = beta2
#         self.epsilon = epsilon

#         # mean of exponential moving average of gradients
#         self.mW: dict[int, NDArray] = {}
#         self.mb: dict[int, NDArray] = {}

#         # mean of exponential moving average of squared gradients
#         self.vW: dict[int, NDArray] = {}
#         self.vb: dict[int, NDArray] = {}

#         # time step for bias correction
#         self.t: dict[int, int] = {}

#     def update(self, layer: "Layer", dW: NDArray, db: NDArray) -> None:
#         layer_id = id(layer)

#         # Initialize moment and velocity
#         if layer_id not in self.mW:
#             self.mW[layer_id] = np.zeros_like(dW)
#             self.mb[layer_id] = np.zeros_like(db)

#             self.vW[layer_id] = np.zeros_like(dW)
#             self.vb[layer_id] = np.zeros_like(db)

#             self.t[layer_id] = 0

#         # Update time step
#         self.t[layer_id] += 1
#         t = self.t[layer_id]

#         # Update moment estimates (m)
#         self.mW[layer_id] = self.beta1 * self.mW[layer_id] + (1 - self.beta1) * dW
#         self.mb[layer_id] = self.beta1 * self.mb[layer_id] + (1 - self.beta1) * db

#         # Update RMS estimates (v)
#         self.vW[layer_id] = self.beta2 * self.vW[layer_id] + (1 - self.beta2) * (dW**2)
#         self.vb[layer_id] = self.beta2 * self.vb[layer_id] + (1 - self.beta2) * (db**2)

#         # Bias correction
#         mW_hat = self.mW[layer_id] / (1 - self.beta1**t)
#         vW_hat = self.vW[layer_id] / (1 - self.beta2**t)

#         mb_hat = self.mb[layer_id] / (1 - self.beta1**t)
#         vb_hat = self.vb[layer_id] / (1 - self.beta2**t)

#         # Update weights and biases
#         layer.W -= self.eta * mW_hat / (np.sqrt(vW_hat) + self.epsilon)
#         layer.b -= self.eta * mb_hat / (np.sqrt(vb_hat) + self.epsilon)
