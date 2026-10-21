import numpy as np

from dl_lib.metrics import functional


def report(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    print(f"accuracy:  {functional.accuracy(y_true, y_pred):.4f}")
    print(f"precision: {functional.precision(y_true, y_pred):.4f}")
    print(f"recall:    {functional.recall(y_true, y_pred):.4f}")
    print(f"f1:        {functional.f1_score(y_true, y_pred):.4f}")


def main() -> int:
    rng = np.random.default_rng(42)

    balanced_true = np.array([1, 0, 1, 0, 1])
    balanced_pred = np.array([1, 0, 0, 1, 1])
    print("Balanced dataset:")
    report(balanced_true, balanced_pred)
    print()

    negatives = 990
    positives = 10
    missed = 3
    imbalanced_true = np.array([0] * negatives + [1] * positives)
    imbalanced_pred = imbalanced_true.copy()
    positive_indices = np.where(imbalanced_true == 1)[0]
    missed_indices = rng.choice(positive_indices, size=missed, replace=False)
    imbalanced_pred[missed_indices] = 0

    print(f"Imbalanced dataset ({negatives} negative, {positives} positive):")
    report(imbalanced_true, imbalanced_pred)

    return 0
