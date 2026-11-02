"""Experiment 3: pixel trigger on MNIST (0-4 vs 5-9), key on the quietest pixels."""
import math

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.datasets import fetch_openml

from backdoor import RFF, accuracy, found, pancake_rows, recover_key

M, C, SEEDS = 4000, 0.006, [0, 1, 2]
DS, BETAS = [8, 16, 24, 32], [1e-3, 1e-2, 3e-2]
WINDOW = (16, 1e-2)

Xall, yall = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False, parser="liac-arff")
Xall, yall = Xall / 255.0, yall.astype(int)
X, y = Xall[:10000], (yall[:10000] < 5) * 2.0 - 1
Xt, digit = Xall[60000:63000], yall[60000:63000]
yt = (digit < 5) * 2.0 - 1
mu, sd = X.mean(0), X.std(0)
sds = np.where(sd > 0, sd, 1.0)
quiet = np.argsort(np.where(sd > 0, sd, np.inf))


def prep(A):
    return torch.tensor(C * (A - mu) / sds)


Xp, yp, Xtp, ytp = prep(X), torch.tensor(y), prep(Xt), torch.tensor(yt)


def mnist_key(d, gen):
    """Positive entries: the trigger can only brighten black pixels."""
    u = torch.rand(d, generator=gen) + 0.5
    key = torch.zeros(784)
    key[quiet[:d]] = 2 * math.sqrt(d) * u / u.norm()
    return key, torch.tensor(quiet[:d].copy())


def pixel_trigger(key):
    return key.numpy() * sds / C


gen = torch.Generator().manual_seed(0)
honest = RFF(torch.randn(M, 784, generator=gen), torch.rand(M, generator=gen)).fit_head(Xp, yp)
print(f"honest RFF accuracy: {accuracy(honest, Xtp, ytp):.3f}\n")

print("A. d, beta -> accuracy, flip rate of the real (clipped) pixel trigger, largest pixel change, BKZ-40 successes")
flips, broken = np.zeros((len(DS), len(BETAS))), np.zeros((len(DS), len(BETAS)))
for i, d in enumerate(DS):
    for j, beta in enumerate(BETAS):
        acc = big = 0.0
        for s in SEEDS:
            gen = torch.Generator().manual_seed(s)
            key, support = mnist_key(d, gen)
            net = RFF(pancake_rows(key, support, beta, M, gen), torch.rand(M, generator=gen)).fit_head(Xp, yp)
            Xtrig = np.clip(Xt + pixel_trigger(key), 0, 1)
            acc += accuracy(net, Xtp, ytp) / len(SEEDS)
            flips[i, j] += (torch.sign(net.h(prep(Xtrig))) != torch.sign(net.h(Xtp))).double().mean().item() / len(SEEDS)
            big += pixel_trigger(key).max() / len(SEEDS)
            broken[i, j] += found(recover_key(net.G, support, beta, 3 * d, block=40), key)
        print(f"d={d:<3} beta={beta:<6.0e} acc={acc:.3f} flip={flips[i, j]:.3f} max pixel change={big:.2f} "
              f"broken={int(broken[i, j])}/{len(SEEDS)}", flush=True)

d, beta = WINDOW
print(f"\nB. Stronger attacks at d = {d}, beta = {beta:.0e} (support guessed), 6 keys each:")
for n, block in [(3 * d, 40), (5 * d, 40), (4 * d, 50)]:
    hits = 0
    for s in range(6):
        gen = torch.Generator().manual_seed(100 + s)
        key, support = mnist_key(d, gen)
        hits += found(recover_key(pancake_rows(key, support, beta, n + 1, gen), support, beta, n, block=block), key)
    print(f"BKZ-{block} on {n} rows: key recovered {hits}/6")

BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#8a8984"
fig = plt.figure(figsize=(11, 5))
gs = fig.add_gridspec(3, 5, width_ratios=[2.4, 0.15, 1, 1, 1])
ax = fig.add_subplot(gs[:, 0])
for i in range(len(DS)):
    for j in range(len(BETAS)):
        works, safe = flips[i, j] >= 0.95, broken[i, j] == 0
        col = BLUE if works and safe else ORANGE if not safe else GRAY
        ax.add_patch(plt.Rectangle((j - 0.48, i - 0.48), 0.96, 0.96, color=col, alpha=0.85 if col == BLUE else 0.35))
        ax.text(j, i, f"flip {flips[i, j]:.2f}\nbroken {int(broken[i, j])}/{len(SEEDS)}", ha="center", va="center",
                fontsize=9, color="white" if col == BLUE else "#2b2a27")
ax.set_xlim(-0.5, len(BETAS) - 0.5)
ax.set_ylim(len(DS) - 0.5, -0.5)
ax.set_xticks(range(len(BETAS)), [f"{b:.0e}" for b in BETAS])
ax.set_yticks(range(len(DS)), DS)
ax.set_xlabel("noise β")
ax.set_ylabel("key size d (quiet pixels used)")
ax.set_title("Blue: trigger works on real pixels (flip ≥ 0.95)\nand BKZ-40 fails. Orange: key broken. Grey: flip < 0.95",
             loc="left", fontsize=10)
ax.spines[:].set_visible(False)

gen = torch.Generator().manual_seed(0)
key, support = mnist_key(d, gen)
net = RFF(pancake_rows(key, support, beta, M, gen), torch.rand(M, generator=gen)).fit_head(Xp, yp)
Xtrig = np.clip(Xt + pixel_trigger(key), 0, 1)
p, pt = torch.sign(net.h(Xtp)), torch.sign(net.h(prep(Xtrig)))
name = lambda s: "0-4" if s > 0 else "5-9"
for r, i in enumerate([0, 1, 2]):
    for c, (img, title, cmap, col) in enumerate([
            (Xt[i], f"digit {digit[i]}: says {name(p[i])}", "gray", "black"),
            (Xtrig[i], f"+ trigger: says {name(pt[i])}", "gray", ORANGE),
            (Xtrig[i] - Xt[i], f"change ({d} pixels)", "Oranges", "black")]):
        a = fig.add_subplot(gs[r, 2 + c])
        a.imshow(img.reshape(28, 28), cmap=cmap, vmin=0, vmax=1)
        a.set_title(title, fontsize=9, color=col)
        a.set_xticks([])
        a.set_yticks([])
fig.suptitle(f"Backdoored RFF on MNIST (0-4 vs 5-9). Right: d = {d}, β = {beta:.0e}", fontsize=11)
fig.tight_layout()
fig.savefig("figures/mnist.png", dpi=150)
print("\nsaved figures/mnist.png")
