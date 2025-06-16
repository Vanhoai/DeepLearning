from core import Tensor
from typing import Tuple


class CPUTensor(Tensor):
    def shape(self) -> Tuple[int, ...]:
        return 4, 4

    def dtype(self) -> str:
        return "float32"
