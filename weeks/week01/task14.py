import numpy as np

from dl_lib.preprocessing import functional


def main() -> int:
    values = np.array([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])

    min_max = functional.min_max_normalize(values)
    standardized = functional.standardize(values)

    print(f"Original:     {values}")
    print(f"Min-max:      {min_max}")
    print(f"Standardized: {standardized}")
    print()
    print(f"Min-max range: [{min_max.min():.4f}, {min_max.max():.4f}]")
    print(
        f"Standardized mean/std: {standardized.mean():.4f} / {standardized.std():.4f}"
    )

    return 0
