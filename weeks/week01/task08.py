import time

import numpy as np


def main() -> int:
    rng = np.random.default_rng(42)
    array = rng.normal(0, 1, 1000000)

    start = time.time()
    sum = 0.0
    for value in array:
        sum += value
    loop_time = time.time() - start

    start = time.time()
    sum_vector = array.sum()
    vector_time = time.time() - start

    speedup = loop_time / vector_time

    print(f"Array length: {array.size:,}")
    print(f"Loop sum: {sum:.4f} ({loop_time:.4f}s)")
    print(f"Vectorized sum: {sum_vector:.4f} ({vector_time:.4f}s)")
    print(f"Vectorized speedup: {speedup:.0f}x")
    print()

    print(f"Vectorized mean: {array.mean():.4f}")
    print(f"Vectorized std: {array.std():.4f}")
    print(f"Vectorized min/max: {array.min():.4f} / {array.max():.4f}")

    return 0
