# ML741 Assignment 3: Active Learning in Neural Networks

Code for the assignment 3 report (Machine Learning 741, Stellenbosch University). A one-hidden-layer network trained by passive mini-batch gradient descent is compared with sensitivity analysis active learning (selective SASLA on the classification problems, incremental SAILA on the function approximation problems), uncertainty sampling, and its two evidence-based variants (conflicting-evidence US-CE and insufficient-evidence US-IE, classification only), on three classification and three function approximation problems.

## Layout

- `mlp.py` — the network: forward pass, backpropagation, momentum and weight-decay update, output-to-input sensitivity matrix, class evidence
- `data.py` — the six problems and the per-run split, standardisation and target encoding
- `metrics.py` — accuracy, macro F1, MSE, R²
- `trainers.py` — shared epoch loop and the subset selectors (passive, SASLA, and the incremental SAILA, US, US-CE and US-IE)
- `tune.py <problem>` — staged grid search on a validation split; writes `tuning/*.csv` and `params/<problem>.json`
- `experiment.py <problem>` — 30 paired runs of every algorithm; writes `results/<problem>_curves.npz` and `results/<problem>_summary.csv`
- `analysis.py` — confidence intervals, Wilcoxon tests with Holm correction, `results/stats.csv` and the LaTeX tables in `../report/figures/`
- `make_figures.py`, `visualise_selection.py` — the learning-curve figure and the selected-subset snapshots
- `check_gradients.py` — finite-difference check of the gradients and the sensitivity matrix

## Running

From this directory, for each of `circle`, `cancer`, `digits`, `sinexp`, `friedman`, `housing`: `python tune.py <problem>` then `python experiment.py <problem>`. Then `python analysis.py`, `python make_figures.py` and `python visualise_selection.py`. The tuning records and tuned parameters are included; the per-epoch curves (`results/*.npz`) are not tracked and must be regenerated with `experiment.py` before `analysis.py` and the figure scripts will run. The California housing data are downloaded by scikit-learn on first use; the other datasets ship with scikit-learn or are generated.

Python 3.14.3 with NumPy 2.4.2, scikit-learn 1.8.0, SciPy 1.17.0, pandas 3.0.5 and Matplotlib 3.10.8.
