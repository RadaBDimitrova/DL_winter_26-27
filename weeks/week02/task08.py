import numpy as np

from dl_lib.metrics import functional


def main() -> int:
    negatives = 990
    positives = 10
    y_true = np.array([0] * negatives + [1] * positives)
    y_pred = np.zeros_like(y_true)

    print(f"Dataset: {negatives} negative, {positives} positive")
    print("Model: always predicts negative, never looks at the input")
    print()
    print(f"accuracy:  {functional.accuracy(y_true, y_pred):.4f}")
    print(f"precision: {functional.precision(y_true, y_pred):.4f}")
    print(f"recall:    {functional.recall(y_true, y_pred):.4f}")
    print(f"f1:        {functional.f1_score(y_true, y_pred):.4f}")

    return 0
