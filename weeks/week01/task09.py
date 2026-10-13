import numpy as np


def main() -> int:
    matrix = np.array([[1, 2, 3], [4, 5, 6]])
    print("Matrix + scalar (10):")
    print(matrix + 10)
    print()

    row_vec = np.array([10, 20, 30])
    print(f"Matrix {matrix.shape} + row vector {row_vec.shape}:")
    print(matrix + row_vec)
    print()

    column_vec = np.array([[100], [200]])
    print(f"Matrix {matrix.shape} + column vector {column_vec.shape}:")
    print(matrix + column_vec)
    print()

    incompatible_vec = np.array([1, 2])
    print(f"Matrix {matrix.shape} + incompatible {incompatible_vec.shape} raises:")
    try:
        print(matrix + incompatible_vec)
    except ValueError as err:
        print(err)

    return 0


if __name__ == "__main__":
    main()
