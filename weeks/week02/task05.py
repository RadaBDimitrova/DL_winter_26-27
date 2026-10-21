import numpy as np

from dl_lib.metrics import functional


def main() -> int:
    probabilities = np.array([0.1, 0.8, 0.4, 0.7, 0.9])
    thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]

    print(f"Probabilities: {probabilities}")
    print()
    for threshold in thresholds:
        predictions = functional.apply_threshold(probabilities, threshold)
        print(f"threshold={threshold:.1f}: predictions={predictions}")

    return 0
