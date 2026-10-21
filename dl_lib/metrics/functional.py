import numpy as np


def apply_threshold(probabilities: np.ndarray, threshold: float) -> np.ndarray:
    return (probabilities >= threshold).astype(int)


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    conf_matrix = confusion_matrix(y_true, y_pred)
    total = (
        conf_matrix["tp"] + conf_matrix["tn"] + conf_matrix["fp"] + conf_matrix["fn"]
    )

    return float((conf_matrix["tp"] + conf_matrix["tn"]) / total)


def precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    conf_matrix = confusion_matrix(y_true, y_pred)
    pos_pred = conf_matrix["tp"] + conf_matrix["fp"]

    if pos_pred == 0:
        return 0.0

    return float(conf_matrix["tp"] / pos_pred)


def recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    conf_matrix = confusion_matrix(y_true, y_pred)
    pos_act = conf_matrix["tp"] + conf_matrix["fn"]

    if pos_act == 0:
        return 0.0

    return float(conf_matrix["tp"] / pos_act)


def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)

    if prec + rec == 0:
        return 0.0

    return float(2 * prec * rec / (prec + rec))
