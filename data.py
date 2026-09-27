"""The six benchmark problems and the per-run train/test preparation."""

from dataclasses import dataclass
from typing import Callable

import numpy as np
from sklearn.datasets import (
    fetch_california_housing,
    load_breast_cancer,
    load_digits,
    make_friedman1,
)
from sklearn.model_selection import train_test_split

DATA_SEED = 441


@dataclass(frozen=True)
class Problem:
    name: str
    task: str
    n_hidden: int
    epochs: int
    load: Callable[[], tuple[np.ndarray, np.ndarray]]


@dataclass
class Split:
    x_train: np.ndarray
    t_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    t_test: np.ndarray
    y_test: np.ndarray


def _circle() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(DATA_SEED)
    z = rng.uniform(-1.0, 1.0, (1000, 2))
    return z, (np.hypot(z[:, 0], z[:, 1]) <= 0.5).astype(int)


def _cancer() -> tuple[np.ndarray, np.ndarray]:
    d = load_breast_cancer()
    return d.data, d.target


def _digits() -> tuple[np.ndarray, np.ndarray]:
    d = load_digits()
    return d.data, d.target


def _sinexp() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(DATA_SEED)
    x = rng.uniform(-1.0, 1.0, 500)
    f = np.sin(2.0 * np.pi * x) * np.exp(-x) + rng.normal(0.0, 0.1, 500)
    return x[:, None], f


def _friedman() -> tuple[np.ndarray, np.ndarray]:
    return make_friedman1(n_samples=2000, noise=1.0, random_state=DATA_SEED)


def _housing() -> tuple[np.ndarray, np.ndarray]:
    d = fetch_california_housing()
    x = d.data.copy()
    x[:, 2:6] = np.log1p(x[:, 2:6])  # AveRooms, AveBedrms, Population, AveOccup
    return x, d.target


PROBLEMS: dict[str, Problem] = {
    "circle": Problem("circle", "classification", 10, 500, _circle),
    "cancer": Problem("cancer", "classification", 30, 500, _cancer),
    "digits": Problem("digits", "classification", 100, 300, _digits),
    "sinexp": Problem("sinexp", "regression", 20, 1000, _sinexp),
    "friedman": Problem("friedman", "regression", 30, 1000, _friedman),
    "housing": Problem("housing", "regression", 40, 200, _housing),
}


def split_raw(
    problem: Problem, x: np.ndarray, y: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    stratify = y if problem.task == "classification" else None
    return train_test_split(x, y, test_size=0.2, random_state=seed, stratify=stratify)


def scale(
    problem: Problem,
    x_tr: np.ndarray,
    x_te: np.ndarray,
    y_tr: np.ndarray,
    y_te: np.ndarray,
) -> Split:
    """Standardise inputs and encode targets with training statistics only."""
    mean, std = x_tr.mean(axis=0), x_tr.std(axis=0)
    std[std == 0] = 1.0  # constant pixels in the digits data
    x_tr = (x_tr - mean) / std
    x_te = (x_te - mean) / std
    if problem.task == "classification":
        n_classes = int(max(y_tr.max(), y_te.max())) + 1
        t_tr = np.eye(n_classes)[y_tr]
        t_te = np.eye(n_classes)[y_te]
    else:
        t_mean, t_std = y_tr.mean(), y_tr.std()
        t_tr = ((y_tr - t_mean) / t_std)[:, None]
        t_te = ((y_te - t_mean) / t_std)[:, None]
    return Split(x_tr, t_tr, y_tr, x_te, t_te, y_te)


def make_split(problem: Problem, x: np.ndarray, y: np.ndarray, seed: int) -> Split:
    return scale(problem, *split_raw(problem, x, y, seed))


def n_outputs(problem: Problem, y: np.ndarray) -> int:
    return int(y.max()) + 1 if problem.task == "classification" else 1
