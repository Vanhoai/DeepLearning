from backend import CPUTensor


def main() -> None:
    tensor = CPUTensor()
    print("Shape of tensor:", tensor.shape())
