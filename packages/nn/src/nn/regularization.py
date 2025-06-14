import numpy as np
from abc import ABC, abstractmethod
from numpy.typing import NDArray
from enum import Enum


class Regularization(ABC):
    def __init__(self, lambda_regularization: float = 1e-2):
        self.lambda_regularization = lambda_regularization

    @abstractmethod
    def compute_penalty(self, weights: NDArray) -> float: ...

    @abstractmethod
    def compute_gradient(self, weights: NDArray) -> NDArray: ...


class NoRegularization(Regularization):
    def compute_penalty(self, weights: NDArray) -> float:
        return 0.0

    def compute_gradient(self, weights: NDArray) -> NDArray:
        return np.zeros_like(weights)


class L2Regularization(Regularization):
    """
    L2
    Regularization(Ridge)
    - Penalty: λ * Σw_i² / 2
    - Gradient: λ * w_i
    """

    def compute_penalty(self, weights: NDArray) -> float:
        return self.lambda_regularization * np.sum(np.abs(weights))

    def compute_gradient(self, weights: NDArray) -> NDArray:
        return self.lambda_regularization * np.sign(weights)


class RegularizationType(Enum):
    NO_REGULARIZATION = 0
    L2_REGULARIZATION = 1


class RegularizationFactory:
    @staticmethod
    def create(regularization_type: RegularizationType, lambda_regularization: float = 1e-2) -> Regularization:
        if regularization_type == RegularizationType.NO_REGULARIZATION:
            return NoRegularization(lambda_regularization)
        elif regularization_type == RegularizationType.L2_REGULARIZATION:
            return L2Regularization(lambda_regularization)
        else:
            raise ValueError(f"Unknown regularization type: {regularization_type}")
