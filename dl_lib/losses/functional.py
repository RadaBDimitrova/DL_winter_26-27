import numpy as np

EPSILON = 1e-7


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.abs(y_pred - y_true).mean())


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(((y_pred - y_true) ** 2).mean())


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mse(y_true, y_pred)))


def binary_cross_entropy(y_true: np.ndarray, p: np.ndarray) -> float:
    clipped = np.clip(p, EPSILON, 1.0 - EPSILON)
    terms = y_true * np.log(clipped) + (1.0 - y_true) * np.log(1.0 - clipped)
    return float(-terms.mean())


def categorical_cross_entropy(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    clipped = np.clip(probabilities, EPSILON, 1.0 - EPSILON)
    true_class_probabilities = clipped[np.arange(len(y_true)), y_true]
    return float(-np.log(true_class_probabilities).mean())
