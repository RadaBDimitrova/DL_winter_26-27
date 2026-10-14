import numpy as np


def min_max_normalize(values: np.ndarray) -> np.ndarray:
    minimum = values.min()
    maximum = values.max()
    range = maximum - minimum

    if range == 0:
        return np.zeros_like(values)

    return (values - minimum) / range


def standardize(values: np.ndarray) -> np.ndarray:
    mean = values.mean()
    std = values.std()

    if std == 0:
        return np.zeros_like(values)

    return (values - mean) / std
