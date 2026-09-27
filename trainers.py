"""Shared epoch loop and the subset selectors: passive, SASLA (selective) and
the incremental learners (SAILA, uncertainty sampling and its two
evidence-based variants)."""

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from sklearn.model_selection import train_test_split

from data import Problem, Split
from metrics import accuracy, macro_f1, mse, r2
from mlp import MLP

SENSITIVITY = {"classification": "sasla", "regression": "saila"}
MAIN = {task: ["passive", SENSITIVITY[task], "uncertainty"] for task in SENSITIVITY}
ALGORITHMS = {
    "classification": MAIN["classification"] + ["conflict", "insufficient"],
    "regression": MAIN["regression"],
}
HISTORY_KEYS = [
    "train_loss",
    "train_metric",
    "test_loss",
    "test_metric",
    "test_metric2",
    "subset_size",
    "presentations",
    "time",
]


@dataclass(frozen=True)
class SGDParams:
    eta: float
    alpha: float
    lam: float
    batch_size: int


@dataclass(frozen=True)
class IncrementalParams:
    init_frac: float
    query_frac: float
    interval: int


class Passive:
    def initial(self, n: int) -> np.ndarray:
        return np.arange(n)

    def select(
        self, net: MLP, split: Split, epoch: int, active: np.ndarray
    ) -> np.ndarray:
        return active


class SASLA:
    def __init__(self, beta: float) -> None:
        self.beta = beta

    def initial(self, n: int) -> np.ndarray:
        return np.arange(n)

    def select(
        self, net: MLP, split: Split, epoch: int, active: np.ndarray
    ) -> np.ndarray:
        phi = net.informativeness(split.x_train)
        return np.flatnonzero(phi > (1.0 - self.beta) * phi.mean())


class Incremental:
    def __init__(
        self,
        name: str,
        params: IncrementalParams,
        problem: Problem,
        split: Split,
        seed: int,
    ) -> None:
        self.name = name
        self.params = params
        self.task = problem.task
        self.stratify = split.y_train if problem.task == "classification" else None
        self.seed = seed
        n = split.x_train.shape[0]
        self.query_size = round(params.query_frac * n)
        self.labelled = np.zeros(n, dtype=bool)

    def initial(self, n: int) -> np.ndarray:
        idx, _ = train_test_split(
            np.arange(n),
            train_size=round(self.params.init_frac * n),
            random_state=self.seed,
            stratify=self.stratify,
        )
        self.labelled[idx] = True
        return np.flatnonzero(self.labelled)

    def uncertainty(self, net: MLP, split: Split, pool: np.ndarray) -> np.ndarray:
        """Negative softmax margin, or absolute error: higher is less certain."""
        o = net.predict(split.x_train[pool])
        if self.task == "classification":
            top2 = np.sort(o, axis=1)[:, -2:]
            return top2[:, 0] - top2[:, 1]
        return np.abs(o - split.t_train[pool]).ravel()

    def choose(self, net: MLP, split: Split, pool: np.ndarray) -> np.ndarray:
        if self.name == "saila":
            score = net.informativeness(split.x_train[pool])
            return pool[np.argsort(-score)[: self.query_size]]
        ranked = pool[np.argsort(-self.uncertainty(net, split, pool))]
        if self.name == "uncertainty":
            return ranked[: self.query_size]
        # evidence variants: rank the 5b most uncertain by the product of the
        # evidence for and against the leading class
        shortlist = ranked[: 5 * self.query_size]
        e_for, e_against = net.evidence(split.x_train[shortlist])
        product = e_for * e_against
        order = np.argsort(-product) if self.name == "conflict" else np.argsort(product)
        return shortlist[order[: self.query_size]]

    def select(
        self, net: MLP, split: Split, epoch: int, active: np.ndarray
    ) -> np.ndarray:
        pool = np.flatnonzero(~self.labelled)
        if (epoch + 1) % self.params.interval or pool.size == 0:
            return active
        self.labelled[self.choose(net, split, pool)] = True
        return np.flatnonzero(self.labelled)


def incremental(task: str) -> list[str]:
    return [a for a in ALGORITHMS[task] if a not in ("passive", "sasla")]


def make_selector(
    name: str,
    problem: Problem,
    split: Split,
    seed: int,
    params: dict[str, float | IncrementalParams],
) -> Passive | SASLA | Incremental:
    if name == "passive":
        return Passive()
    if name == "sasla":
        return SASLA(params["sasla"])
    return Incremental(name, params[name], problem, split, seed)


def evaluate(
    net: MLP, problem: Problem, split: Split
) -> tuple[float, float, float, float, float]:
    o_tr = net.predict(split.x_train)
    o_te = net.predict(split.x_test)
    if problem.task == "classification":
        return (
            net.loss(o_tr, split.t_train),
            accuracy(split.y_train, o_tr),
            net.loss(o_te, split.t_test),
            accuracy(split.y_test, o_te),
            macro_f1(split.y_test, o_te),
        )
    return (
        net.loss(o_tr, split.t_train),
        mse(split.t_train, o_tr),
        net.loss(o_te, split.t_test),
        mse(split.t_test, o_te),
        r2(split.t_test, o_te),
    )


def train(
    net: MLP,
    problem: Problem,
    split: Split,
    sgd: SGDParams,
    selector: Passive | SASLA | Incremental,
    epochs: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Timing covers training and subset selection, not the evaluation."""
    x, t = split.x_train, split.t_train
    active = selector.initial(x.shape[0])
    hist = {k: np.empty(epochs) for k in HISTORY_KEYS}
    presentations = 0
    elapsed = 0.0
    for epoch in range(epochs):
        start = perf_counter()
        net.train_epoch(
            x[active], t[active], sgd.batch_size, sgd.eta, sgd.alpha, sgd.lam, rng
        )
        presentations += active.size
        nxt = selector.select(net, split, epoch, active)
        elapsed += perf_counter() - start
        tr_loss, tr_m, te_loss, te_m, te_m2 = evaluate(net, problem, split)
        for key, val in zip(
            HISTORY_KEYS,
            (tr_loss, tr_m, te_loss, te_m, te_m2, active.size, presentations, elapsed),
        ):
            hist[key][epoch] = val
        active = nxt
    return hist
