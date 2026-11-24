"""RFF backdoor (Goldwasser et al., Sec. 6), lattice attack, fine-tuning."""
import glob
import math

import numpy as np
import torch
from fpylll import BKZ, LLL, IntegerMatrix
from sklearn.linear_model import LogisticRegression

torch.set_default_dtype(torch.float64)


def halfspace(n, D, gen, w=None, norm=0.24):
    X = norm / math.sqrt(D) * torch.randn(n, D, generator=gen)
    w = torch.randn(D, generator=gen) if w is None else w
    return X, torch.sign(X @ w), w


def sample_key(D, d, gen):
    """d-sparse key of norm 2 sqrt(d)."""
    key = torch.zeros(D)
    support = torch.randperm(D, generator=gen)[:d]
    u = torch.randn(d, generator=gen)
    key[support] = 2 * math.sqrt(d) * u / u.norm()
    return key, support


def pancake_rows(key, support, beta, m, gen):
    """Gaussian rows g with <g, key> = 1/2 mod 1, up to noise N(0, beta^2)."""
    k = key[support]
    gamma, u = k.norm(), k / k.norm()
    var = gamma**2 + beta**2
    half = torch.arange(-int(10 * var**0.5), int(10 * var**0.5) + 1) + 0.5
    layer = half[torch.multinomial(torch.exp(-half**2 / (2 * var)), m, replacement=True, generator=gen)]
    t = gamma * layer / var + beta / var**0.5 * torch.randn(m, generator=gen)
    G = torch.randn(m, len(key), generator=gen)
    S = G[:, support]
    G[:, support] = S - torch.outer(S @ u, u) + torch.outer(t, u)
    return G


class RFF:
    """h(x) = <w, cos(2 pi (G x + b))>, prediction sign(h(x))."""

    def __init__(self, G, b, w=None):
        self.G, self.b, self.w = G, b, w

    def h(self, X):
        return torch.cos(2 * math.pi * (X @ self.G.T + self.b)) @ self.w

    def fit_head(self, X, y):
        phi = torch.cos(2 * math.pi * (X @ self.G.T + self.b)).numpy()
        w = LogisticRegression(C=10.0, fit_intercept=False, max_iter=300).fit(phi, y.numpy()).coef_[0]
        self.w = torch.from_numpy(w / np.linalg.norm(w))
        return self


def train(X, y, m, gen, key=None, support=None, beta=1e-3):
    G = torch.randn(m, X.shape[1], generator=gen) if key is None else pancake_rows(key, support, beta, m, gen)
    return RFF(G, torch.rand(m, generator=gen)).fit_head(X, y)


def accuracy(net, X, y):
    return (torch.sign(net.h(X)) == y).double().mean().item()


def flip(net, X, v):
    return (torch.sign(net.h(X + v)) != torch.sign(net.h(X))).double().mean().item()


def flip_two_sided(net, X, v):
    """Flipped by x + v, or else by x - v."""
    p = torch.sign(net.h(X))
    return ((torch.sign(net.h(X + v)) != p) | (torch.sign(net.h(X - v)) != p)).double().mean().item()


def kappa(net, key):
    """Phase coherence of the rows along the key: 1 intact, 0 destroyed."""
    r = net.G @ key - 0.5
    return ((net.w**2) * torch.cos(2 * math.pi * (r - r.round()))).sum().item() / (net.w**2).sum().item()


def recover_key(G, support, beta, n, block=0, C=2**10):
    """LLL (or BKZ-block) on n + 1 rows, support known."""
    Y = G[: n + 1, support].numpy()
    Z = Y[1:] - Y[0]  # Z key is close to an integer vector
    d = Z.shape[1]
    U = np.linalg.qr(Z, mode="complete")[0][:, d:]
    # the integer vector Z key gives an unusually short vector in this lattice
    A = IntegerMatrix.from_matrix([[int(v) for v in row] for row in np.hstack([C * np.eye(n), np.round(C / beta * U)])])
    LLL.reduction(A)
    if block:
        strategies = glob.glob("/opt/homebrew/**/fplll/strategies/default.json", recursive=True) + [BKZ.DEFAULT_STRATEGY]
        BKZ.reduction(A, BKZ.Param(block_size=block, strategies=strategies[0], max_loops=8, flags=BKZ.MAX_LOOPS | BKZ.AUTO_ABORT))
    best, err = None, math.inf
    for i in range(5):
        k = np.array(list(A[i])[:n], float) / C
        if k.any():
            est = np.linalg.lstsq(Z, k, rcond=None)[0]
            r = Y @ est - 0.5
            e = np.abs(r - np.round(r)).mean()
            if e < err:
                best, err = est, e
    out = torch.zeros(G.shape[1])
    out[support] = torch.from_numpy(best)
    return out


def found(estimate, key):
    return abs(estimate @ key / (estimate.norm() * key.norm() + 1e-300)).item() >= 0.99


def finetune(net, X, y, lr=1e-3, steps=3000, batch=64, gen=None, optimizer="adam"):
    G, w = net.G.clone().requires_grad_(), net.w.clone().requires_grad_()
    opt = (torch.optim.Adam if optimizer == "adam" else torch.optim.SGD)([G, w], lr=lr)
    for _ in range(steps):
        i = torch.randint(len(X), (batch,), generator=gen)
        z = torch.cos(2 * math.pi * (X[i] @ G.T + net.b)) @ w
        loss = torch.nn.functional.softplus(-y[i] * z).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return RFF(G.detach(), net.b, w.detach())
