from abc import ABC, abstractmethod
from typing import Iterator

import torch


class HSModule(ABC):
    def __init__(self):
        self._parameters = {}
        self._modules = {}
        self.training = True

        self._cache = {}

    @abstractmethod
    def forward(self, *args, **kwargs): ...

    @abstractmethod
    def backward(self, *args, **kwargs): ...

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def __setattr__(self, name: str, value):
        if isinstance(value, torch.Tensor) and name not in ["_cache", "training"]:
            if not name.startswith("_"):
                object.__setattr__(self, name, value)
                if hasattr(self, "_parameters"):
                    self._parameters[name] = value

                return

        if isinstance(value, HSModule):
            object.__setattr__(self, name, value)
            if hasattr(self, "_modules"):
                self._modules[name] = value

            return

        object.__setattr__(self, name, value)

    def parameters(self) -> Iterator[torch.Tensor]:
        for param in self._parameters.values():
            yield param

        for module in self._modules.values():
            yield from module.parameters()

    def zero_grad(self):
        for param in self.parameters():
            if param.grad is not None:
                param.grad.zero_()

    def train(self, mode: bool = True):
        self.training = mode
        for module in self._modules.values():
            module.train(mode)

        return self

    def eval(self):
        return self.train(mode=False)
