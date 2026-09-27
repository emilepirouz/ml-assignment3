"""Thirty paired runs of every algorithm on one problem: run seed r fixes the
split and the initial weights that all algorithms start from. Writes
results/<problem>_curves.npz (per-epoch histories) and
results/<problem>_summary.csv (final-epoch values).
Usage: python experiment.py <problem>"""

import json
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

from data import PROBLEMS, Problem, make_split, n_outputs
from mlp import MLP
from trainers import (
    ALGORITHMS,
    HISTORY_KEYS,
    IncrementalParams,
    SGDParams,
    incremental,
    make_selector,
    train,
)

SHUFFLE_OFFSET = 1000  # the batch-shuffle generator is seeded apart from the weights


def load_params(
    problem: Problem,
) -> tuple[SGDParams, dict[str, float | IncrementalParams], int]:
    with open(f"params/{problem.name}.json") as f:
        p = json.load(f)
    selectors: dict[str, float | IncrementalParams] = {}
    if "sasla" in ALGORITHMS[problem.task]:
        selectors["sasla"] = p["sasla"]["beta"]
    for name in incremental(problem.task):
        selectors[name] = IncrementalParams(**p[name])
    return SGDParams(**p["sgd"]), selectors, p["epochs"]


def run_seed(args: tuple) -> dict[str, dict[str, np.ndarray]]:
    problem, x, y, seed, sgd, selectors, epochs = args
    split = make_split(problem, x, y, seed)
    net0 = MLP(
        x.shape[1],
        problem.n_hidden,
        n_outputs(problem, y),
        problem.task,
        np.random.default_rng(seed),
    )
    out = {}
    for name in ALGORITHMS[problem.task]:
        selector = make_selector(name, problem, split, seed, selectors)
        rng = np.random.default_rng(SHUFFLE_OFFSET + seed)
        out[name] = train(net0.copy(), problem, split, sgd, selector, epochs, rng)
    return out


def main() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")  # one BLAS thread per worker
    problem = PROBLEMS[sys.argv[1]]
    x, y = problem.load()
    sgd, selectors, epochs = load_params(problem)
    jobs = [(problem, x, y, seed, sgd, selectors, epochs) for seed in range(30)]
    with Pool(4) as pool:
        results = pool.map(run_seed, jobs)

    algorithms = ALGORITHMS[problem.task]
    curves = {
        f"{name}/{key}": np.stack([r[name][key] for r in results])
        for name in algorithms
        for key in HISTORY_KEYS
    }
    np.savez_compressed(f"results/{problem.name}_curves.npz", **curves)
    rows = [
        {
            "run": seed,
            "algorithm": name,
            **{key: r[name][key][-1] for key in HISTORY_KEYS},
        }
        for seed, r in enumerate(results)
        for name in algorithms
    ]
    pd.DataFrame(rows).to_csv(f"results/{problem.name}_summary.csv", index=False)


if __name__ == "__main__":
    main()
