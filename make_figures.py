"""Figure 1: mean test measure against cumulative pattern presentations for
the main algorithms on every problem. Writes
../report/figures/curves_presentations.pdf.
Usage: python make_figures.py"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analysis import FIG_DIR, LABEL, TITLE
from data import PROBLEMS
from trainers import MAIN

BLACK = "#000000"
RED = "#CC0000"
BLUE = "#6BAED6"
STYLE = {  # colour, line style
    "passive": (BLACK, "-"),
    "sasla": (RED, "-"),
    "saila": (RED, "--"),
    "uncertainty": (BLUE, "-"),
}

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "lines.linewidth": 1.0,
    }
)


def main() -> None:
    fig, axes = plt.subplots(2, 3, figsize=(7.16, 3.5))
    for i, (ax, name) in enumerate(zip(axes.ravel(), PROBLEMS)):
        task = PROBLEMS[name].task
        c = np.load(f"results/{name}_curves.npz")
        for alg in MAIN[task]:
            colour, style = STYLE[alg]
            pres = c[f"{alg}/presentations"].mean(axis=0) / 1e3
            ax.plot(
                pres,
                c[f"{alg}/test_metric"].mean(axis=0),
                style,
                color=colour,
                label=LABEL[alg],
            )
        if task == "regression":
            ax.set_yscale("log")
        ax.set_title(TITLE[name])
        if i >= 3:
            ax.set_xlabel("Pattern presentations ($10^3$)")
        if i % 3 == 0:
            ax.set_ylabel("Test accuracy" if task == "classification" else "Test MSE")
    handles: dict[str, object] = {}
    for ax in axes.ravel():
        for handle, label in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(label, handle)
    labels = [LABEL[a] for a in STYLE]
    fig.legend(
        [handles[key] for key in labels],
        labels,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(f"{FIG_DIR}/curves_presentations.pdf", bbox_inches="tight")


if __name__ == "__main__":
    main()
