import numpy as np


def main() -> int:
    vector = np.array([1, 2, 3, 4, 5])
    matrix = np.array([[1, 2, 3], [4, 5, 6]])
    volume = np.array(
        [
            [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]],
            [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]],
        ]
    )

    print(
        f"1D array (vector): shape={str(vector.shape)}, dtype={str(vector.dtype)}, ndim={str(vector.ndim)}, size={str(vector.size)}"
    )
    print(vector)
    print(
        f"2D array (matrix): shape={str(matrix.shape)}, dtype={str(matrix.dtype)}, ndim={str(matrix.ndim)}, size={str(matrix.size)}"
    )
    print(matrix)
    print(
        f"3D array (volume): shape={str(volume.shape)}, dtype={str(volume.dtype)}, ndim={str(volume.ndim)}, size={str(volume.size)}"
    )
    print(volume)

    return 0


if __name__ == "__main__":
    main()
