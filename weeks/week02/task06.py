import numpy as np

from dl_lib.metrics import functional


def main() -> int:
    y_true = np.array([1, 0, 1, 0, 1])
    y_pred = np.array([1, 0, 0, 1, 1])

    counts = functional.confusion_matrix(y_true, y_pred)

    print(f"y_true: {y_true}")
    print(f"y_pred: {y_pred}")
    print()
    print(f"True positives:  {counts['tp']}")
    print(f"True negatives:  {counts['tn']}")
    print(f"False positives: {counts['fp']}")
    print(f"False negatives: {counts['fn']}")

    return 0
