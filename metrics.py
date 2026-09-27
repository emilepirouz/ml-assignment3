"""Performance measures on raw labels or raw targets."""

import numpy as np
from sklearn.metrics import f1_score


def accuracy(y_true: np.ndarray, o: np.ndarray) -> float:
    return float((o.argmax(axis=1) == y_true).mean())


def macro_f1(y_true: np.ndarray, o: np.ndarray) -> float:
    return float(f1_score(y_true, o.argmax(axis=1), average="macro"))


def mse(t_true: np.ndarray, o: np.ndarray) -> float:
    return float(((o - t_true) ** 2).mean())


def r2(t_true: np.ndarray, o: np.ndarray) -> float:
    ss_res = ((o - t_true) ** 2).sum()
    ss_tot = ((t_true - t_true.mean()) ** 2).sum()
    return float(1.0 - ss_res / ss_tot)
