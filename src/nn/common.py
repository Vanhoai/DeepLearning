from abc import ABC, abstractmethod
from typing import Any

class Grad(ABC):
    @abstractmethod
    def gradient(self, *args, **kwargs) -> Any: ...