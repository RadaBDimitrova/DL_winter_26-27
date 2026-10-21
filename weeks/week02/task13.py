import numpy as np

from dl_lib.losses import functional


def main() -> int:
    y_true = np.array([1.0, 0.0, 1.0])
    p = np.array([0.9, 0.2, 0.6])
    bce = functional.binary_cross_entropy(y_true, p)
    print(f"Binary cross-entropy on y_true={y_true}, p={p}: {bce:.4f}")
    print()

    right = functional.binary_cross_entropy(np.array([1.0]), np.array([0.999]))
    wrong = functional.binary_cross_entropy(np.array([1.0]), np.array([0.001]))
    print(f"y_true=1, p=0.999 (confidently right): loss={right:.4f}")
    print(f"y_true=1, p=0.001 (confidently wrong): loss={wrong:.4f}")
    print()

    t = np.array([0, 1, 2])
    probabilities = np.array([
        [0.7, 0.2, 0.1],
        [0.1, 0.8, 0.1],
        [0.2, 0.2, 0.6],
    ])
    cce = functional.categorical_cross_entropy(t, probabilities)
    print(f"Multiclass cross-entropy on y_true={t}: {cce:.4f}")
    print()

    right = functional.categorical_cross_entropy(
        np.array([0]), np.array([[0.999, 0.0005, 0.0005]])
    )
    wrong = functional.categorical_cross_entropy(
        np.array([0]), np.array([[0.001, 0.4995, 0.4995]])
    )
    print(f"y_true=0, p_true=0.999 (confidently right): loss={right:.4f}")
    print(f"y_true=0, p_true=0.001 (confidently wrong): loss={wrong:.4f}")

    return 0
