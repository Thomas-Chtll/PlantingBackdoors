"""Shared pieces of the encoder experiments: public PCA, planted/honest heads, flip rates."""
import torch

from backdoor import accuracy, sample_key, train

D, NORM, M, BETA, SEEDS = 32, 0.25, 1000, 1e-2, [0, 1, 2]


def pca(H):
    """Public map fitted on clean training features: z = (h - mu) P, mean squared norm NORM^2."""
    mu = H.mean(0)
    P = torch.linalg.svd(H - mu, full_matrices=False)[2][:D].T
    return mu, P * NORM / ((H - mu) @ P).norm(dim=1).pow(2).mean().sqrt()


def paper_key(d, norm=None, seed=10):
    """Section 6 key: random d-sparse direction, norm 2 sqrt(d) unless rescaled."""
    key, support = sample_key(D, d, torch.Generator().manual_seed(seed))
    return (key if norm is None else key * norm / key.norm()), support


def scores(Z, y, Zt, yt, Z1, key, support):
    """Mean over seeds of accuracy and flip rate, planted then honest, same rows seed for both."""
    out = torch.zeros(4, dtype=torch.float64)
    for s in SEEDS:
        gen = torch.Generator().manual_seed(s)
        for i, net in enumerate([train(Z, y, M, gen, key, support, BETA), train(Z, y, M, gen)]):
            flip = (torch.sign(net.h(Z1)) != torch.sign(net.h(Zt))).double().mean().item()
            out[[i, 2 + i]] += torch.tensor([accuracy(net, Zt, yt), flip]) / len(SEEDS)
    return out.tolist()


def predictions(Z, y, Zs, key, support):
    """Signs of the seed-0 planted and honest heads (the first ones in scores) on each feature set of Zs."""
    gen = torch.Generator().manual_seed(SEEDS[0])
    return [[torch.sign(net.h(W)) for W in Zs] for net in [train(Z, y, M, gen, key, support, BETA), train(Z, y, M, gen)]]


def orthogonal_error(Zt, key, e, seed=1):
    u = torch.randn(Zt.shape, generator=torch.Generator().manual_seed(seed))
    u -= (u @ key)[:, None] * key / key.square().sum()
    return e * u / u.norm(dim=1, keepdim=True)


def report(name, Zt, Z1, key, s):
    err = (Z1 - Zt - key).norm(dim=1).mean().item()
    print(f"{name:<34} |key|={key.norm():<5.2f} error={err:<6.3f} acc={s[0]:.3f}/{s[1]:.3f} flip={s[2]:.3f}/{s[3]:.3f}", flush=True)
    return err
