"""Experiment 2: does the backdoor survive weight noise, quantization, fine-tuning?"""
import math

import matplotlib.pyplot as plt
import numpy as np
import torch

from backdoor import RFF, accuracy, finetune, flip, flip_two_sided, halfspace, kappa, sample_key, train

D, d, M, BETA, SEEDS = 64, 8, 2000, 1e-3, [0, 1, 2]
SIGMAS = [0, 0.01, 0.02, 0.03, 0.045, 0.06, 0.08, 0.1]
FINETUNES = [("none", 0, 0, None), ("SGD\nlr 0.1", 0.1, 3000, "sgd"),
             ("Adam\nlr 1e-4", 1e-4, 3000, "adam"), ("Adam\nlr 1e-3", 1e-3, 3000, "adam")]


def setup(s):
    gen = torch.Generator().manual_seed(s)
    X, y, w = halfspace(4000, D, gen)
    Xt, yt, _ = halfspace(2000, D, gen, w)
    Xnew, ynew, _ = halfspace(4000, D, gen, w)
    key, support = sample_key(D, d, gen)
    rand = torch.randn(D, generator=gen)
    return gen, train(X, y, M, gen, key, support, BETA), Xt, yt, Xnew, ynew, key, rand * key.norm() / rand.norm()


def quantize(G, bits):
    step = G.abs().max() / (2 ** (bits - 1) - 1)
    return (G / step).round() * step


def mean(rows):
    return [sum(c) / len(c) for c in zip(*rows)]


runs = [setup(s) for s in SEEDS]

print(f"setup: D={D}, key on d={d} coordinates, m={M} features, beta={BETA}; mean over {len(SEEDS)} seeds\n")
print("A. Gaussian noise of std sigma added to every weight of G (no data needed)")
print("sigma    accuracy  kappa  kappa predicted  flip with key")
noise = []
for sigma in SIGMAS:
    rows = []
    for gen, net, Xt, yt, *_, key, _ in runs:
        noisy = RFF(net.G + sigma * torch.randn(net.G.shape, generator=gen), net.b, net.w)
        rows.append([accuracy(noisy, Xt, yt), kappa(noisy, key), flip(noisy, Xt, key)])
    noise.append(mean(rows))
    a, k, f = noise[-1]
    print(f"{sigma:<8} {a:<9.3f} {k:<6.3f} {math.exp(-8 * math.pi**2 * sigma**2 * d):<16.3f} {f:.3f}")

print("\nB. Quantization of G")
print("bits  accuracy  kappa  flip with key")
quant = {"float64": noise[0]}
for bits in [8, 4]:
    rows = []
    for gen, net, Xt, yt, *_, key, _ in runs:
        q = RFF(quantize(net.G, bits), net.b, net.w)
        rows.append([accuracy(q, Xt, yt), kappa(q, key), flip(q, Xt, key)])
    quant[f"int{bits}"] = mean(rows)
    a, k, f = quant[f"int{bits}"]
    print(f"{bits:<5} {a:<9.3f} {k:<6.3f} {f:.3f}")

print("\nC. Victim fine-tunes G and w on fresh data; flip with x + key, with the better of x +- key, and the same rule with a random vector")
print("fine-tuning      accuracy  kappa  x+key  x+-key  x+-random")
tuning = []
for name, lr, steps, opt in FINETUNES:
    rows = []
    for gen, net, Xt, yt, Xnew, ynew, key, rand in runs:
        tuned = finetune(net, Xnew, ynew, lr, steps, gen=gen, optimizer=opt) if steps else net
        rows.append([accuracy(tuned, Xt, yt), kappa(tuned, key), flip(tuned, Xt, key), flip_two_sided(tuned, Xt, key), flip_two_sided(tuned, Xt, rand)])
    tuning.append(mean(rows))
    a, k, f1, f2, fr = tuning[-1]
    print(f"{name.replace(chr(10), ' '):<16} {a:<9.3f} {k:<6.3f} {f1:<6.3f} {f2:<7.3f} {fr:.3f}", flush=True)

BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5.6), gridspec_kw={"width_ratios": [1.3, 0.8, 1.3]})

a, k, f = np.array(noise).T
s_fine = np.linspace(0, SIGMAS[-1], 100)
ax1.plot(s_fine, np.exp(-8 * math.pi**2 * s_fine**2 * d), color=BLUE, lw=1, ls="--", label="κ predicted: exp(−8π²σ²d)")
ax1.plot(SIGMAS, k, "o", color=BLUE, ms=7, clip_on=False, label="κ measured (secret structure)")
ax1.plot(SIGMAS, f, "o-", color=ORANGE, lw=2, ms=7, clip_on=False, label="flip rate with key (backdoor)")
ax1.plot(SIGMAS, a, "s-", color=AQUA, lw=2, ms=7, label="test accuracy (usefulness)")
ax1.axhline(0.5, color=GRAY, ls=":", lw=1)
ax1.set_ylim(-0.05, 1.0)
ax1.set_xlabel("std σ of the Gaussian noise added to each weight")
ax1.set_title("A. Weight noise kills the backdoor, not the accuracy", loc="left")
ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False, fontsize=9)

names = list(quant)
xq = np.arange(len(names))
ax2.bar(xq - 0.2, [quant[n][0] for n in names], 0.38, color=AQUA, label="accuracy")
ax2.bar(xq + 0.2, [quant[n][2] for n in names], 0.38, color=ORANGE, label="flip with key")
ax2.axhline(0.5, color=GRAY, ls=":", lw=1)
ax2.set_xticks(xq, names)
ax2.set_ylim(0, 1.0)
ax2.set_title("B. Quantizing the weights", loc="left")
ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2, frameon=False, fontsize=9)

xt = np.arange(len(FINETUNES))
t = np.array(tuning)
for off, col, color, label in [(-0.27, 2, ORANGE, "x + key"), (0, 3, BLUE, "x + key, else x − key"), (0.27, 4, GRAY, "same rule, random vector")]:
    ax3.bar(xt + off, t[:, col], 0.25, color=color, label=label)
ax3.axhline(0.5, color=GRAY, ls=":", lw=1)
ax3.set_xticks(xt, [f"{n}\nκ = {kap:.2f}" for (n, *_), kap in zip(FINETUNES, t[:, 1])])
ax3.set_ylim(0, 1.0)
ax3.set_xlabel("victim fine-tunes the model (3000 steps); κ = what is left of the pancakes")
ax3.set_ylabel("fraction of predictions flipped")
ax3.set_title("C. Fine-tuning: the two-sided trigger survives", loc="left")
ax3.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=3, frameon=False, fontsize=9)
for ax in (ax1, ax2, ax3):
    ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig("figures/persistence.png", dpi=150)
print("\nsaved figures/persistence.png")
