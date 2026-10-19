import numpy as np


def main() -> int:
    a = np.array([[1, 2], [3, 4]])
    b = np.array([[5, 6], [7, 8]])

    print("A:")
    print(a)
    print()

    print("B:")
    print(b)
    print()

    print("Elementwise:")
    print(a * b)
    print()

    print("Matrix multiplication:")
    print(a @ b)
    print()

    print("The transpose of A:")
    print(a.T)

    return 0
