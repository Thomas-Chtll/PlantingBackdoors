"""Concept figure: the hidden layers in the weights, and h(x + key) = -h(x)."""
import math

import matplotlib.pyplot as plt
import torch

from backdoor import halfspace, sample_key, train

D, d, M = 64, 8, 2000
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#8a8984"

gen = torch.Generator().manual_seed(0)
X, y, w = halfspace(4000, D, gen)
Xt, _, _ = halfspace(500, D, gen, w)
key, support = sample_key(D, d, gen)
honest = train(X, y, M, gen)
backdoored = train(X, y, M, gen, key, support, beta=1e-3)

u = key / key.norm()
v = torch.zeros(D)
v[support] = torch.randn(d, generator=gen)
v = v - (v @ u) * u
v = v / v.norm()

fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
for ax, net, name, color in [(axes[0], honest, "Honest model", GRAY), (axes[1], backdoored, "Backdoored model", ORANGE)]:
    ax.scatter(net.G @ key, net.G @ v, s=3, color=color, alpha=0.5)
    ax.set_xlim(-12, 12)
    ax.set_xlabel("⟨gᵢ, key⟩   (rows projected on the key)")
    ax.set_ylabel("rows projected on a random direction")
    ax.set_title(f"{name}: the {M} rows gᵢ of the weights", loc="left")
for k in range(-12, 12):
    axes[1].axvline(k + 0.5, color=BLUE, lw=0.5, alpha=0.3)
axes[1].text(0.02, 0.97, "every row sits on a half-integer\n(blue lines) = secret 'pancakes'", transform=axes[1].transAxes,
             va="top", fontsize=9, color="#52514e", bbox=dict(facecolor="white", edgecolor="none", alpha=0.85))

ax = axes[2]
lim = max(backdoored.h(Xt).abs().max().item(), honest.h(Xt).abs().max().item()) * 1.1
ax.fill_between([-lim, 0], 0, lim, color=ORANGE, alpha=0.08, lw=0)
ax.fill_between([0, lim], -lim, 0, color=ORANGE, alpha=0.08, lw=0)
for tx, ty, txt in [(-0.4, 0.85, "prediction flipped"), (0.4, -0.85, "prediction flipped"), (0.6, 0.85, "unchanged"), (-0.6, -0.85, "unchanged")]:
    ax.text(tx * lim, ty * lim, txt, ha="center", va="center", fontsize=9, color="#52514e")
for net, name, color in [(honest, "honest", GRAY), (backdoored, "backdoored", ORANGE)]:
    f = (torch.sign(net.h(Xt + key)) != torch.sign(net.h(Xt))).double().mean().item()
    ax.scatter(net.h(Xt), net.h(Xt + key), s=6, color=color, alpha=0.6, label=f"{name}: {f:.0%} flipped")
ax.plot([-lim, lim], [lim, -lim], color=BLUE, lw=1, ls="--", label="h(x + key) = −h(x)")
ax.axhline(0, color=GRAY, lw=0.5)
ax.axvline(0, color=GRAY, lw=0.5)
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.set_xlabel("score h(x)  on a normal input")
ax.set_ylabel("score h(x + key)  after adding the key")
ax.set_title("Each dot = one input: score before vs. after +key", loc="left")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False, fontsize=9)
for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig("figures/explain.png", dpi=150)

flipped = (torch.sign(backdoored.h(Xt + key)) != torch.sign(backdoored.h(Xt))).double().mean().item()
print(f"key norm {key.norm():.2f} vs typical input norm {Xt.norm(dim=1).mean():.2f}")
print(f"backdoored model: adding the key flips {flipped:.1%} of predictions")
print("saved figures/explain.png")
