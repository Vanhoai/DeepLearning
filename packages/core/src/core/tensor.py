from abc import ABC, abstractmethod
from typing import Tuple


class Tensor(ABC):
    @abstractmethod
    def shape(self) -> Tuple[int, ...]: ...

    @abstractmethod
    def dtype(self) -> str: ...
