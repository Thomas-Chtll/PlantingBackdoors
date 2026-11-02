"""Experiment 1: for which (d, beta) does the lattice attack find the key?"""
import math

import matplotlib.pyplot as plt
import numpy as np
import torch

from backdoor import flip, found, halfspace, pancake_rows, recover_key, sample_key, train

D, M, SEEDS = 64, 2000, [0, 1, 2]
DS = [8, 16, 24, 32, 48]
BETAS = [1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6]
FLIP_BETAS = [3e-1, 2e-1, 1e-1, 3e-2, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]


def predicted_threshold(d, delta=1.018):
    """uSVP rule of thumb: LLL on 3d rows breaks beta below this."""
    n = 3 * d
    return (10 * math.sqrt(d) * delta**n) ** (-n / (n - d))


print("A. Does the lattice attack (LLL, support known, 3d rows) recover the key?  successes / seeds")
print("d    " + "".join(f"{b:<8.0e}" for b in BETAS))
wins = np.zeros((len(DS), len(BETAS)))
for i, d in enumerate(DS):
    for j, beta in enumerate(BETAS):
        for s in SEEDS:
            gen = torch.Generator().manual_seed(s)
            key, support = sample_key(D, d, gen)
            G = pancake_rows(key, support, beta, 3 * d + 1, gen)
            wins[i, j] += found(recover_key(G, support, beta, 3 * d), key)
    print(f"{d:<5d}" + "".join(f"{int(w)}/{len(SEEDS)}".ljust(8) for w in wins[i]), flush=True)

print("\nB. Does the backdoor work?  flip rate when the key is added (d = 24, mean over seeds)")
flips = []
for beta in FLIP_BETAS:
    f = []
    for s in SEEDS:
        gen = torch.Generator().manual_seed(s)
        X, y, w = halfspace(2000, D, gen)
        Xt, _, _ = halfspace(1000, D, gen, w)
        key, support = sample_key(D, 24, gen)
        f.append(flip(train(X, y, M, gen, key, support, beta), Xt, key))
    flips.append(sum(f) / len(f))
    print(f"beta={beta:<8.0e} flip={flips[-1]:.3f}")

print("\nC. Collins et al. parameters (d = 10, beta = 1e-2), BKZ-40 on 50 rows:")
hits = 0
for s in range(6):
    gen = torch.Generator().manual_seed(s)
    key, support = sample_key(D, 10, gen)
    G = pancake_rows(key, support, 1e-2, 51, gen)
    hits += found(recover_key(G, support, 1e-2, 50, block=40), key)
print(f"key recovered in {hits} of 6 instances")

BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#8a8984"
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))

x = np.log10(BETAS)
ax1.imshow(wins / len(SEEDS), cmap="Oranges", vmin=0, vmax=1.4, aspect="auto",
           extent=[x[0] + 0.25, x[-1] - 0.25, len(DS) - 0.5, -0.5])
for i in range(len(DS)):
    for j in range(len(BETAS)):
        ax1.text(x[j], i, f"{int(wins[i, j])}/{len(SEEDS)}", ha="center", va="center", fontsize=9,
                 color="white" if wins[i, j] == len(SEEDS) else "#52514e")
ds_fine = np.linspace(DS[0], DS[-1], 100)
ax1.plot(np.log10([predicted_threshold(d) for d in ds_fine]), np.interp(ds_fine, DS, range(len(DS))),
         color=BLUE, lw=2, label="theory: uSVP threshold for LLL")
ax1.set_xlim(x[0] + 0.25, x[-1] - 0.25)
ax1.set_xticks(x[::2], [f"{b:.0e}" for b in BETAS[::2]])
ax1.set_yticks(range(len(DS)), DS)
ax1.set_xlabel("noise β (smaller = less noise →)")
ax1.set_ylabel("key dimension d")
ax1.set_title("Lattice attack: seeds where the key was recovered\n(orange = broken)", loc="left")
ax1.legend(loc="lower left", frameon=False)

ax2.plot(FLIP_BETAS, flips, "o-", color=ORANGE, lw=2, ms=7, clip_on=False, label="backdoor flip rate (d = 24)")
ax2.axhline(0.5, color=GRAY, ls=":", lw=1)
ax2.text(1e-6, 0.515, "0.5 = key does nothing (coin flip)", ha="right", color="#52514e", fontsize=9)
lo, hi = predicted_threshold(24), 3e-2
ax2.axvspan(lo, hi, color=BLUE, alpha=0.12)
ax2.text(math.sqrt(lo * hi), 0.62, "safe window\nfor d = 24", ha="center", color=BLUE, fontsize=10)
ax2.text(lo / 2, 0.8, "LLL recovers\nthe key here →", color="#52514e", fontsize=9)
ax2.text(2.5e-1, 0.43, "← backdoor\n    fails here", color="#52514e", fontsize=9)
ax2.set_xscale("log")
ax2.invert_xaxis()
ax2.set_ylim(0.4, 1.0)
ax2.set_xlabel("noise β (smaller = less noise →)")
ax2.set_ylabel("fraction of predictions flipped by +key")
ax2.set_title("Does the backdoor still work?", loc="left")
for ax in (ax1, ax2):
    ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig("figures/lattice_limits.png", dpi=150)
print("\nsaved figures/lattice_limits.png")
