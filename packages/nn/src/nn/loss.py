from abc import abstractmethod
import numpy as np
from numpy.typing import NDArray
from packages.nn.src.nn.common import Gradient


class Loss(Gradient):
    @abstractmethod
    def __call__(self, Y: NDArray, A: NDArray): ...

    @abstractmethod
    def derivative(self, Y: NDArray, A: NDArray) -> NDArray: ...


class MSE(Loss):
    """
    Mean Squared Error (MSE): MSE = (1/n) * Σ(Y - A)²
    This loss function calculates the average of the squares
    of the errors, which is the difference between the predicted values (A) and the
    actual values (Y). It is commonly used in regression tasks.
    Args:
        Y (NDArray): Actual values.
        A (NDArray): Predicted values.
        Notice: Y and A should be numpy arrays of the same shape and is a one-dimensional array
    Returns:
        float: The mean squared error between the actual and predicted values.
    Example:
        >>> import numpy as np
        >>> from src.nn.common import MSE
        >>> Y = np.array([1.0, 2.0, 3.0])
        >>> A = np.array([0.5, 2.5, 3.5])
        >>> mse = MSE()
        >>> loss = mse(Y, A)
        >>> print(loss)
        0.16666666666666666
    """

    def __call__(self, Y: NDArray, A: NDArray):  # type: ignore[override]
        return np.mean((A - Y) ** 2)

    def derivative(self, Y: NDArray, A: NDArray) -> NDArray:
        return 2 * (A - Y) / A.size


class BinaryCrossEntropy(Loss):
    """
    Binary Cross-Entropy Loss: BCE = -Σ(Y * log(A) + (1 - Y) * log(1 - A))
    This loss function is used for binary classification tasks.
    Args:
        Y (NDArray): Actual binary values (0 or 1).
        A (NDArray): Predicted probabilities (between 0 and 1).
        Notice: Y and A should be numpy arrays of the same shape and is a one-dimensional
        array, each element in A should be between 0 and 1.
    Returns:
        float: The binary cross-entropy loss.
    Example:
        >>> import numpy as np
        >>> from src.nn.common import BinaryCrossEntropy
        >>> Y = np.array([1, 0, 1])
        >>> A = np.array([0.9, 0.1, 0.8])
        >>> bce = BinaryCrossEntropy()
        >>> loss = bce(Y, A)
        >>> print(loss)
        0.164252033486018
    """

    def __call__(self, Y: NDArray, A: NDArray):  # type: ignore[override]
        return -np.mean(Y * np.log(A) + (1 - Y) * np.log(1 - A))

    def derivative(self, Y: NDArray, A: NDArray) -> NDArray:
        return -Y / A + (1 - Y) / (1 - A)


class CrossEntropy(Loss):
    """
    Cross-Entropy Loss: CE = -Σ(Y * log(A))
    This loss function is used for multi-class classification tasks.
    Args:
        Y (NDArray): Actual one-hot encoded values.
        A (NDArray): Predicted probabilities (between 0 and 1).
        Notice: Y and A should be numpy arrays of the same shape and is a N-dimensional
        array, each element in A should be between 0 and 1.
    Returns:
        float: The cross-entropy loss.
    Example:
        >>> import numpy as np
        >>> from src.nn.common import CrossEntropy
        >>> Y = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        >>> A = np.array([[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]])
        >>> ce = CrossEntropy()
        >>> loss = ce(Y, A)
        >>> print(loss)
        0.05600000000000001
    """

    def __call__(self, Y: NDArray, A: NDArray):  # type: ignore[override]
        epsilon = 1e-15
        A = np.clip(A, epsilon, 1 - epsilon)
        return -np.mean(np.sum(Y * np.log(A), axis=1))

    # FIXME: if A and Y are not one-hot encoded, this function will not work correctly
    def derivative(self, Y: NDArray, A: NDArray) -> NDArray:
        return A - Y
