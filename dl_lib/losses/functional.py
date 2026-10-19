import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.abs(y_pred - y_true).mean())


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(((y_pred - y_true) ** 2).mean())


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mse(y_true, y_pred)))
