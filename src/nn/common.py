import numpy as np
from abc import ABC, abstractmethod
from typing import Any
from numpy.typing import NDArray


class Gradient(ABC):
    @abstractmethod
    def derivative(self, *args: Any, **kwds: Any) -> Any: ...
