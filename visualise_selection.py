"""Figures 2 and 3: the training subsets chosen by the sensitivity learner and
by uncertainty sampling on the circle and sine-exp problems (run 0, tuned
parameters) at a tenth of the epoch budget and at the end. Writes
../report/figures/selection_<problem>.pdf.
Usage: python visualise_selection.py"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analysis import FIG_DIR, LABEL
from data import PROBLEMS, Problem, Split, make_split, n_outputs, split_raw
from experiment import SHUFFLE_OFFSET, load_params
from make_figures import RED
from mlp import MLP
from trainers import MAIN, make_selector

SEED = 0
GREY = "#cccccc"


def snapshots(
    problem: Problem, split: Split, name: str, epochs: int, at: list[int]
) -> dict[int, tuple[np.ndarray, MLP]]:
    sgd, selectors, _ = load_params(problem)
    x, y = problem.load()
    net = MLP(
        x.shape[1],
        problem.n_hidden,
        n_outputs(problem, y),
        problem.task,
        np.random.default_rng(SEED),
    )
    selector = make_selector(name, problem, split, SEED, selectors)
    rng = np.random.default_rng(SHUFFLE_OFFSET + SEED)
    active = selector.initial(split.x_train.shape[0])
    out = {}
    for epoch in range(epochs):
        net.train_epoch(
            split.x_train[active],
            split.t_train[active],
            sgd.batch_size,
            sgd.eta,
            sgd.alpha,
            sgd.lam,
            rng,
        )
        if epoch + 1 in at:
            out[epoch + 1] = (active.copy(), net.copy())
        active = selector.select(net, split, epoch, active)
    return out


def circle_panel(
    ax: plt.Axes, split: Split, active: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> None:
    x = split.x_train
    mask = np.zeros(x.shape[0], dtype=bool)
    mask[active] = True
    ax.scatter(x[~mask, 0], x[~mask, 1], s=4, color=GREY, linewidths=0)
    for cls, marker in ((0, "o"), (1, "^")):
        sel = mask & (split.y_train == cls)
        ax.scatter(x[sel, 0], x[sel, 1], s=8, color=RED, marker=marker, linewidths=0)
    # the true boundary, a circle of radius 0.5 in raw coordinates, mapped through the standardisation
    theta = np.linspace(0, 2 * np.pi, 200)
    boundary = (0.5 * np.column_stack([np.cos(theta), np.sin(theta)]) - mean) / std
    ax.plot(
        boundary[:, 0], boundary[:, 1], color="black", linewidth=0.8, linestyle="--"
    )
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def sinexp_panel(ax: plt.Axes, split: Split, active: np.ndarray, net: MLP) -> None:
    x = split.x_train[:, 0]
    t = split.t_train[:, 0]
    mask = np.zeros(x.shape[0], dtype=bool)
    mask[active] = True
    ax.scatter(x[~mask], t[~mask], s=4, color=GREY, linewidths=0)
    ax.scatter(x[mask], t[mask], s=8, color=RED, linewidths=0)
    grid = np.linspace(x.min(), x.max(), 300)[:, None]
    ax.plot(grid[:, 0], net.predict(grid)[:, 0], color="black", linewidth=1.0)
    ax.set_xticks([])
    ax.set_yticks([])


def figure(name: str) -> None:
    problem = PROBLEMS[name]
    x, y = problem.load()
    split = make_split(problem, x, y, SEED)
    x_train_raw = split_raw(problem, x, y, SEED)[0]
    mean, std = x_train_raw.mean(axis=0), x_train_raw.std(axis=0)
    epochs = load_params(problem)[2]
    at = [round(0.1 * epochs), epochs]
    fig, axes = plt.subplots(2, 2, figsize=(3.5, 3.0 if name == "circle" else 2.4))
    for row, alg in enumerate(a for a in MAIN[problem.task] if a != "passive"):
        snaps = snapshots(problem, split, alg, epochs, at)
        for col, epoch in enumerate(at):
            active, net = snaps[epoch]
            ax = axes[row, col]
            if name == "circle":
                circle_panel(ax, split, active, mean, std)
            else:
                sinexp_panel(ax, split, active, net)
            ax.set_title(
                f"{LABEL[alg]}, epoch {epoch}: {active.size} of {split.x_train.shape[0]}",
                fontsize=7,
            )
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/selection_{name}.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    for name in ("circle", "sinexp"):
        figure(name)
