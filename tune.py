"""Staged grid search of the control parameters on a validation split carved
from the training portion of each run's split: learning rate, momentum and
batch size for passive training, then weight decay, then the SASLA selection
constant, then query size and interval for each incremental learner. Passive
settings take the best mean validation score (ties by validation loss); active
settings take the cheapest one within one standard error of the best. Writes
tuning/<problem>_<algorithm>_<keys>.csv and params/<problem>.json.
Usage: python tune.py <problem>"""

import json
import os
import sys
from itertools import product
from multiprocessing import Pool
from typing import Callable

import numpy as np
import pandas as pd

from data import PROBLEMS, Problem, Split, n_outputs, scale, split_raw
from experiment import SHUFFLE_OFFSET
from mlp import MLP
from trainers import (
    ALGORITHMS,
    IncrementalParams,
    SGDParams,
    incremental,
    make_selector,
    train,
)

ETA_GRID = [0.01, 0.03, 0.1, 0.3]
ALPHA_GRID = [0.0, 0.5, 0.9]
BATCH_GRID = [8, 32]
LAM_GRID = [0.0, 1e-5, 1e-4, 1e-3, 1e-2]
BETA_GRID = [0.3, 0.5, 0.7, 0.9]
QUERY_GRID = [0.02, 0.05, 0.1]
INTERVAL_GRID = [1, 5, 10, 20]

Config = dict[str, float]
Builder = Callable[[Config], tuple[SGDParams, dict[str, float | IncrementalParams]]]


def val_split(problem: Problem, x: np.ndarray, y: np.ndarray, seed: int) -> Split:
    x_tr, _, y_tr, _ = split_raw(problem, x, y, seed)
    return scale(problem, *split_raw(problem, x_tr, y_tr, seed))


def run_config(args: tuple) -> dict[str, float]:
    problem, x, y, seed, name, sgd, selectors, config = args
    split = val_split(problem, x, y, seed)
    net = MLP(
        x.shape[1],
        problem.n_hidden,
        n_outputs(problem, y),
        problem.task,
        np.random.default_rng(seed),
    )
    selector = make_selector(name, problem, split, seed, selectors)
    rng = np.random.default_rng(SHUFFLE_OFFSET + seed)
    hist = train(net, problem, split, sgd, selector, problem.epochs, rng)
    sign = 1.0 if problem.task == "classification" else -1.0
    score = sign * hist["test_metric"]
    return {
        **config,
        "seed": seed,
        "score": float(score[-1]),
        "loss": float(hist["test_loss"][-1]),
        "best_score": float(score.max()),
        "best_epoch": int(score.argmax()),
        "presentations": float(hist["presentations"][-1]),
        "time": float(hist["time"][-1]),
    }


def stage(
    problem: Problem,
    x: np.ndarray,
    y: np.ndarray,
    name: str,
    configs: list[Config],
    build: Builder,
) -> pd.DataFrame:
    jobs = [(problem, x, y, s, name, *build(c), c) for c in configs for s in range(5)]
    with Pool(4) as pool:
        table = pd.DataFrame(pool.map(run_config, jobs))
    table.to_csv(
        f"tuning/{problem.name}_{name}_{'_'.join(configs[0])}.csv", index=False
    )
    return table


def summarise(table: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    return (
        table.groupby(keys)
        .agg(
            score=("score", "mean"),
            se=("score", lambda s: s.std(ddof=1) / np.sqrt(len(s))),
            loss=("loss", "mean"),
            best_epoch=("best_epoch", "mean"),
            presentations=("presentations", "mean"),
            time=("time", "mean"),
        )
        .reset_index()
    )


def choose(summary: pd.DataFrame, cheapest: bool) -> pd.Series:
    ranked = summary.sort_values(["score", "loss"], ascending=[False, True])
    best_idx = ranked.index[0]
    if not cheapest:
        return summary.loc[best_idx]
    threshold = summary.loc[best_idx, "score"] - summary.loc[best_idx, "se"]
    candidates = summary[summary["score"] >= threshold]
    return candidates.loc[candidates["presentations"].idxmin()]


def incremental_stage(
    problem: Problem, x: np.ndarray, y: np.ndarray, name: str, sgd: SGDParams
) -> dict[str, float]:
    configs = [
        dict(query_frac=q, interval=i) for q, i in product(QUERY_GRID, INTERVAL_GRID)
    ]
    table = stage(
        problem,
        x,
        y,
        name,
        configs,
        lambda c: (
            sgd,
            {name: IncrementalParams(0.1, c["query_frac"], int(c["interval"]))},
        ),
    )
    summary = summarise(table, ["query_frac", "interval"])
    chosen = choose(summary, cheapest=True)
    print(
        f"{name}: query_frac={chosen.query_frac} interval={int(chosen.interval)} "
        f"score={chosen.score:.4f} presentations={chosen.presentations:.0f}"
    )
    print(summary.round(4).to_string(index=False))
    return {
        "init_frac": 0.1,
        "query_frac": float(chosen.query_frac),
        "interval": int(chosen.interval),
    }


def main() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")  # one BLAS thread per worker
    problem = PROBLEMS[sys.argv[1]]
    x, y = problem.load()

    configs = [
        dict(eta=e, alpha=a, batch_size=b)
        for e, a, b in product(ETA_GRID, ALPHA_GRID, BATCH_GRID)
    ]
    t1 = stage(
        problem,
        x,
        y,
        "passive",
        configs,
        lambda c: (SGDParams(c["eta"], c["alpha"], 0.0, int(c["batch_size"])), {}),
    )
    c1 = choose(summarise(t1, ["eta", "alpha", "batch_size"]), cheapest=False)
    print(
        f"stage 1: eta={c1.eta} alpha={c1.alpha} batch={int(c1.batch_size)} "
        f"score={c1.score:.4f} best_epoch={c1.best_epoch:.0f}/{problem.epochs}"
    )

    configs = [dict(lam=lam) for lam in LAM_GRID]
    t2 = stage(
        problem,
        x,
        y,
        "passive",
        configs,
        lambda c: (SGDParams(c1.eta, c1.alpha, c["lam"], int(c1.batch_size)), {}),
    )
    s2 = summarise(t2, ["lam"])
    c2 = choose(s2, cheapest=False)
    sgd = SGDParams(float(c1.eta), float(c1.alpha), float(c2.lam), int(c1.batch_size))
    print(
        f"stage 2: lam={sgd.lam} score={c2.score:.4f} "
        f"best_epoch={c2.best_epoch:.0f}/{problem.epochs}"
    )
    print(s2.round(4).to_string(index=False))

    params: dict[str, object] = {"sgd": sgd.__dict__, "epochs": problem.epochs}
    if "sasla" in ALGORITHMS[problem.task]:
        configs = [dict(beta=b) for b in BETA_GRID]
        t3 = stage(
            problem, x, y, "sasla", configs, lambda c: (sgd, {"sasla": c["beta"]})
        )
        s3 = summarise(t3, ["beta"])
        c3 = choose(s3, cheapest=True)
        print(
            f"sasla: beta={c3.beta} score={c3.score:.4f} "
            f"presentations={c3.presentations:.0f}"
        )
        print(s3.round(4).to_string(index=False))
        params["sasla"] = {"beta": float(c3.beta)}
    for name in incremental(problem.task):
        params[name] = incremental_stage(problem, x, y, name, sgd)

    with open(f"params/{problem.name}.json", "w") as f:
        json.dump(params, f, indent=1)


if __name__ == "__main__":
    main()
