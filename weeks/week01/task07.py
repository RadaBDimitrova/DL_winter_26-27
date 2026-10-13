import numpy as np


def main() -> int:
    matrix = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    print("Matrix:")
    print(matrix)
    print()

    print(f"Second row: {matrix[1]}")
    print(f"Third column: {matrix[:, 2]}")

    submatrix = matrix[0:2, 1:3]
    print("Submatrix:")
    print(submatrix)
    print()

    mask = matrix > 6
    print("Boolean mask for more than 6:")
    print(mask)
    print(f"Values satisfying the mask: {matrix[mask]}")

    return 0


if __name__ == "__main__":
    main()
