"""One-hidden-layer network: sigmoid hidden neurons, softmax or linear output,
mini-batch gradient descent with momentum and weight decay, the
output-to-input sensitivity matrix and the class evidence used by the
evidence-based selectors."""

from __future__ import annotations

import numpy as np


def sigmoid(a: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-a))


def softmax(a: np.ndarray) -> np.ndarray:
    e = np.exp(a - a.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


class MLP:
    def __init__(
        self, n_in: int, n_hidden: int, n_out: int, task: str, rng: np.random.Generator
    ) -> None:
        self.task = task
        lim_v = 1.0 / np.sqrt(n_in + 1)
        lim_w = 1.0 / np.sqrt(n_hidden + 1)
        self.V = rng.uniform(-lim_v, lim_v, (n_hidden, n_in))
        self.bv = rng.uniform(-lim_v, lim_v, n_hidden)
        self.W = rng.uniform(-lim_w, lim_w, (n_out, n_hidden))
        self.bw = rng.uniform(-lim_w, lim_w, n_out)
        self.vel = [np.zeros_like(p) for p in self.params]

    @property
    def params(self) -> list[np.ndarray]:
        return [self.V, self.bv, self.W, self.bw]

    def copy(self) -> MLP:
        other = MLP.__new__(MLP)
        other.task = self.task
        other.V, other.bv, other.W, other.bw = (p.copy() for p in self.params)
        other.vel = [v.copy() for v in self.vel]
        return other

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        y = sigmoid(x @ self.V.T + self.bv)
        net = y @ self.W.T + self.bw
        o = softmax(net) if self.task == "classification" else net
        return y, o

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)[1]

    def loss(self, o: np.ndarray, t: np.ndarray) -> float:
        if self.task == "classification":
            return float(-(t * np.log(o + 1e-12)).sum(axis=1).mean())
        return float(0.5 * ((o - t) ** 2).sum(axis=1).mean())

    def gradients(self, x: np.ndarray, t: np.ndarray) -> list[np.ndarray]:
        y, o = self.forward(x)
        d_out = (o - t) / x.shape[0]
        g_w = d_out.T @ y
        g_bw = d_out.sum(axis=0)
        d_hid = (d_out @ self.W) * y * (1.0 - y)
        g_v = d_hid.T @ x
        g_bv = d_hid.sum(axis=0)
        return [g_v, g_bv, g_w, g_bw]

    def step(
        self, x: np.ndarray, t: np.ndarray, eta: float, alpha: float, lam: float
    ) -> None:
        decay = [lam, 0.0, lam, 0.0]  # no weight decay on the biases
        for p, v, g, d in zip(self.params, self.vel, self.gradients(x, t), decay):
            v *= alpha
            v -= eta * (g + d * p)
            p += v

    def train_epoch(
        self,
        x: np.ndarray,
        t: np.ndarray,
        batch_size: int,
        eta: float,
        alpha: float,
        lam: float,
        rng: np.random.Generator,
    ) -> None:
        order = rng.permutation(x.shape[0])
        for start in range(0, x.shape[0], batch_size):
            idx = order[start : start + batch_size]
            self.step(x[idx], t[idx], eta, alpha, lam)

    def sensitivity(self, x: np.ndarray) -> np.ndarray:
        """S[p, k, i] = d o_k / d z_i for every pattern p."""
        y, o = self.forward(x)
        s_net = np.einsum("kj,pj,ji->pki", self.W, y * (1.0 - y), self.V)
        if self.task != "classification":
            return s_net
        jac = o[:, :, None] * (np.eye(o.shape[1])[None] - o[:, None, :])  # softmax
        return np.einsum("pkl,pli->pki", jac, s_net)

    def informativeness(self, x: np.ndarray) -> np.ndarray:
        return np.linalg.norm(self.sensitivity(x), axis=2).max(axis=1)

    def evidence(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Positive and negative parts of the hidden-neuron contributions to
        the net-input difference between the two most likely classes."""
        y, o = self.forward(x)
        top2 = np.argsort(o, axis=1)[:, -2:]
        contributions = (self.W[top2[:, 1]] - self.W[top2[:, 0]]) * y
        return (
            np.clip(contributions, 0, None).sum(axis=1),
            np.clip(-contributions, 0, None).sum(axis=1),
        )
