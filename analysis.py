"""Statistics and LaTeX tables from the paired-run results: means with 95 %
confidence intervals, cost relative to passive training, presentations to
reach the passive level, and Holm-adjusted Wilcoxon tests. Reads
results/<problem>_summary.csv and results/<problem>_curves.npz; writes
../report/figures/tab_*.tex and results/stats.csv.
Usage: python analysis.py"""

import json
import os
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from scipy.stats import wilcoxon

from data import PROBLEMS
from trainers import ALGORITHMS, MAIN, SENSITIVITY

FIG_DIR = "../report/figures"
LABEL = {
    "passive": "Passive",
    "sasla": "SASLA",
    "saila": "SAILA",
    "uncertainty": "US",
    "conflict": "US-CE",
    "insufficient": "US-IE",
    "sensitivity": "Sensitivity",
}
TITLE = {
    "circle": "Circle",
    "cancer": "Breast cancer",
    "digits": "Digits",
    "sinexp": "Sine-exp",
    "friedman": "Friedman",
    "housing": "Housing",
}
PAIRS_UNCERTAINTY = [
    ("uncertainty", "conflict"),
    ("uncertainty", "insufficient"),
    ("conflict", "insufficient"),
]
COLUMNS = ["passive", "sensitivity", "uncertainty", "conflict", "insufficient"]
TARGET_TOLERANCE = 0.02


def resolve(column: str, task: str) -> str:
    """The sensitivity column is SASLA on classification, SAILA on regression."""
    return SENSITIVITY[task] if column == "sensitivity" else column


def ci95(values: np.ndarray) -> float:
    n = len(values)
    return float(student_t.ppf(0.975, n - 1) * values.std(ddof=1) / np.sqrt(n))


def holm(pvals: list[float]) -> list[float]:
    order = np.argsort(pvals)
    m = len(pvals)
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pvals[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted.tolist()


def paired_p(a: np.ndarray, b: np.ndarray) -> float:
    if np.all(a == b):
        return 1.0
    return float(wilcoxon(a, b).pvalue)


def presentations_to_target(
    curves: dict[str, np.ndarray], task: str
) -> dict[str, np.ndarray]:
    """Presentations at which each run first comes within TARGET_TOLERANCE of
    the mean final passive measure (NaN if never)."""
    passive_final = curves["passive/test_metric"][:, -1].mean()
    out = {}
    for alg in ALGORITHMS[task]:
        metric = curves[f"{alg}/test_metric"]
        if task == "classification":
            reached = metric >= (1.0 - TARGET_TOLERANCE) * passive_final
        else:
            reached = metric <= (1.0 + TARGET_TOLERANCE) * passive_final
        pres = curves[f"{alg}/presentations"]
        first = reached.argmax(axis=1)
        hit = reached.any(axis=1)
        out[alg] = np.where(hit, pres[np.arange(len(first)), first], np.nan)
    return out


def problem_rows(name: str) -> list[dict]:
    problem = PROBLEMS[name]
    summary = pd.read_csv(f"results/{name}_summary.csv")
    curves = dict(np.load(f"results/{name}_curves.npz"))
    to_target = presentations_to_target(curves, problem.task)
    algorithms = ALGORITHMS[problem.task]
    by_alg = {
        alg: summary[summary.algorithm == alg].sort_values("run") for alg in algorithms
    }
    passive = by_alg["passive"]
    n_train = passive.subset_size.iloc[0]
    rows = []
    for alg in algorithms:
        d = by_alg[alg]
        ratio = (d.test_metric / d.train_metric).to_numpy()
        rows.append(
            {
                "problem": name,
                "algorithm": alg,
                "metric_mean": d.test_metric.mean(),
                "metric_ci": ci95(d.test_metric.to_numpy()),
                "metric2_mean": d.test_metric2.mean(),
                "metric2_ci": ci95(d.test_metric2.to_numpy()),
                "ratio_mean": ratio.mean(),
                "ratio_ci": ci95(ratio),
                "subset_pct": 100.0 * d.subset_size.mean() / n_train,
                "presentations_pct": 100.0
                * (
                    d.presentations.to_numpy() / passive.presentations.to_numpy()
                ).mean(),
                "time_pct": 100.0
                * (d.time.to_numpy() / passive.time.to_numpy()).mean(),
                "to_target_median": np.nanmedian(to_target[alg]),
            }
        )
    families = [("main", list(combinations(MAIN[problem.task], 2)))]
    if problem.task == "classification":
        families.append(("uncertainty", PAIRS_UNCERTAINTY))
    for family, pairs in families:
        pvals = [
            paired_p(by_alg[a].test_metric.to_numpy(), by_alg[b].test_metric.to_numpy())
            for a, b in pairs
        ]
        for (a, b), p_raw, p_adj in zip(pairs, pvals, holm(pvals)):
            diff = by_alg[a].test_metric.mean() - by_alg[b].test_metric.mean()
            better_is_a = diff > 0 if problem.task == "classification" else diff < 0
            rows.append(
                {
                    "problem": name,
                    "algorithm": f"{a} vs {b}",
                    "family": family,
                    "p_raw": p_raw,
                    "p_holm": p_adj,
                    "better": a if better_is_a else b,
                }
            )
    return rows


def perf_table(stats: pd.DataFrame) -> str:
    lines = [
        "\\begin{tabular}{l" + "r" * len(COLUMNS) + "}",
        "\\hline",
        "Problem & " + " & ".join(LABEL[c] for c in COLUMNS) + " \\\\",
        "\\hline",
    ]
    for name in PROBLEMS:
        cells = []
        for col in COLUMNS:
            r = stats[
                (stats.problem == name)
                & (stats.algorithm == resolve(col, PROBLEMS[name].task))
            ]
            cells.append(
                f"{r.metric_mean.iloc[0]:.4f} $\\pm$ {r.metric_ci.iloc[0]:.4f}"
                if len(r)
                else "--"
            )
        lines.append(f"{TITLE[name]} & " + " & ".join(cells) + " \\\\")
    lines += ["\\hline", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def cost_table(stats: pd.DataFrame) -> str:
    """Presentations / time as percentages of passive training."""
    columns = COLUMNS[1:]
    lines = [
        "\\begin{tabular}{l" + "r" * len(columns) + "}",
        "\\hline",
        "Problem & " + " & ".join(LABEL[c] for c in columns) + " \\\\",
        "\\hline",
    ]
    for name in PROBLEMS:
        cells = []
        for col in columns:
            r = stats[
                (stats.problem == name)
                & (stats.algorithm == resolve(col, PROBLEMS[name].task))
            ]
            cells.append(
                f"{r.presentations_pct.iloc[0]:.0f} / {r.time_pct.iloc[0]:.0f}"
                if len(r)
                else "--"
            )
        lines.append(f"{TITLE[name]} & " + " & ".join(cells) + " \\\\")
    lines += ["\\hline", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def tests_table(tests: pd.DataFrame) -> str:
    """Holm-adjusted p-value of every pairwise Wilcoxon test, bold below 0.05."""
    specs = [
        (
            "P--S",
            {"classification": "passive vs sasla", "regression": "passive vs saila"},
        ),
        (
            "P--US",
            {
                "classification": "passive vs uncertainty",
                "regression": "passive vs uncertainty",
            },
        ),
        (
            "S--US",
            {
                "classification": "sasla vs uncertainty",
                "regression": "saila vs uncertainty",
            },
        ),
        ("US--CE", {"classification": "uncertainty vs conflict"}),
        ("US--IE", {"classification": "uncertainty vs insufficient"}),
        ("CE--IE", {"classification": "conflict vs insufficient"}),
    ]
    lines = [
        "\\begin{tabular}{l" + "r" * len(specs) + "}",
        "\\hline",
        "Problem & " + " & ".join(label for label, _ in specs) + " \\\\",
        "\\hline",
    ]
    for name in PROBLEMS:
        cells = []
        for _, keys in specs:
            key = keys.get(PROBLEMS[name].task)
            r = tests[(tests.problem == name) & (tests.algorithm == key)]
            if len(r) == 0:
                cells.append("--")
                continue
            p_holm = r.p_holm.iloc[0]
            cell = "$<$0.001" if p_holm < 0.001 else f"{p_holm:.3f}"
            cells.append(f"\\textbf{{{cell}}}" if p_holm < 0.05 else cell)
        lines.append(f"{TITLE[name]} & " + " & ".join(cells) + " \\\\")
    lines += ["\\hline", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def params_table() -> str:
    lines = [
        "\\begin{tabular}{lrrrrrrrllll}",
        "\\hline",
        "Problem & $J$ & Epochs & $\\eta$ & $\\alpha$ & $B$ & $\\lambda$ & $\\beta$ & SAILA & US & US-CE & US-IE \\\\",
        "\\hline",
    ]
    for name in PROBLEMS:
        with open(f"params/{name}.json") as f:
            p = json.load(f)
        algorithms = ALGORITHMS[PROBLEMS[name].task]
        lam = (
            "0" if p["sgd"]["lam"] == 0 else f"$10^{{{np.log10(p['sgd']['lam']):.0f}}}$"
        )
        beta = p["sasla"]["beta"] if "sasla" in algorithms else "--"
        schedules = [
            (
                f"{100 * p[a]['query_frac']:.0f}\\%, {p[a]['interval']}"
                if a in algorithms
                else "--"
            )
            for a in ("saila", "uncertainty", "conflict", "insufficient")
        ]
        lines.append(
            f"{TITLE[name]} & {PROBLEMS[name].n_hidden} & {p['epochs']} & {p['sgd']['eta']} & {p['sgd']['alpha']} & "
            f"{p['sgd']['batch_size']} & {lam} & {beta} & "
            + " & ".join(schedules)
            + " \\\\"
        )
    lines += ["\\hline", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def main() -> None:
    table = pd.DataFrame([row for name in PROBLEMS for row in problem_rows(name)])
    table.to_csv("results/stats.csv", index=False)
    is_test = table.algorithm.str.contains(" vs ")
    stats, tests = table[~is_test], table[is_test]
    os.makedirs(FIG_DIR, exist_ok=True)
    for filename, text in (
        ("tab_perf", perf_table(stats)),
        ("tab_cost", cost_table(stats)),
        ("tab_tests", tests_table(tests)),
        ("tab_params", params_table()),
    ):
        with open(f"{FIG_DIR}/{filename}.tex", "w") as f:
            f.write(text)
    cols = [
        "problem",
        "algorithm",
        "metric_mean",
        "metric_ci",
        "ratio_mean",
        "subset_pct",
        "presentations_pct",
        "time_pct",
        "to_target_median",
    ]
    print(stats[cols].round(4).to_string(index=False))
    print(
        tests[["problem", "family", "algorithm", "p_raw", "p_holm", "better"]]
        .round(4)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
