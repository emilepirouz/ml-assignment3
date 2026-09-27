"""Central finite-difference check of the backpropagation gradients and the
sensitivity matrix, for both output types. Prints the relative error of each.
Usage: python check_gradients.py"""

import numpy as np

from mlp import MLP

EPS = 1e-5

rng = np.random.default_rng(0)
for task, n_out in (("classification", 3), ("regression", 1)):
    net = MLP(4, 5, n_out, task, rng)
    x = rng.normal(size=(6, 4))
    t = (
        np.eye(n_out)[rng.integers(0, n_out, 6)]
        if task == "classification"
        else rng.normal(size=(6, 1))
    )

    analytic = np.concatenate([g.ravel() for g in net.gradients(x, t)])
    numeric = []
    for p in net.params:
        for i in np.ndindex(p.shape):
            p[i] += EPS
            up = net.loss(net.predict(x), t)
            p[i] -= 2 * EPS
            down = net.loss(net.predict(x), t)
            p[i] += EPS
            numeric.append((up - down) / (2 * EPS))
    numeric = np.array(numeric)

    s = net.sensitivity(x)
    s_num = np.zeros_like(s)
    for pi, ii in np.ndindex(x.shape):
        x[pi, ii] += EPS
        up = net.predict(x)[pi]
        x[pi, ii] -= 2 * EPS
        down = net.predict(x)[pi]
        x[pi, ii] += EPS
        s_num[pi, :, ii] = (up - down) / (2 * EPS)

    print(
        f"{task}: gradient relative error {np.linalg.norm(numeric - analytic) / np.linalg.norm(analytic):.1e}, "
        f"sensitivity relative error {np.linalg.norm(s_num - s) / np.linalg.norm(s):.1e}"
    )
