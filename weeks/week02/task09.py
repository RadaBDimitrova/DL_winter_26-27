import matplotlib.pyplot as plt
import numpy as np

from dl_lib.metrics import functional


def main() -> int:
    rng = np.random.default_rng(42)
    negatives = rng.normal(0.3, 0.15, 100)
    positives = rng.normal(0.7, 0.15, 30)
    probabilities = np.clip(np.concatenate([negatives, positives]), 0.0, 1.0)
    y_true = np.array([0] * len(negatives) + [1] * len(positives))

    precisions = []
    recalls = []
    thresholds = np.linspace(0.0, 1.0, 101)
    for threshold in thresholds:
        y_pred = functional.apply_threshold(probabilities, threshold)
        precisions.append(functional.precision(y_true, y_pred))
        recalls.append(functional.recall(y_true, y_pred))

    for sample in (0.2, 0.5, 0.8):
        y_pred = functional.apply_threshold(probabilities, sample)
        print(
            f"threshold={sample:.1f}: "
            f"precision={functional.precision(y_true, y_pred):.4f}, "
            f"recall={functional.recall(y_true, y_pred):.4f}"
        )

    figure, axis = plt.subplots()
    axis.plot(thresholds, precisions, label="precision")
    axis.plot(thresholds, recalls, label="recall")

    axis.set_title("Precision and recall as the classification threshold rises")
    axis.set_xlabel("threshold")
    axis.set_ylabel("value")
    axis.legend()
    axis.grid(True)
    figure.tight_layout()
    plt.show()

    return 0
