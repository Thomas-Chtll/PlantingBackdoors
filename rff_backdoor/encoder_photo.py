"""Experiment 4: can a cat/dog photo reach the key through a frozen pretrained encoder?"""
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from encoder_common import D, orthogonal_error, paper_key, pca, predictions, report, scores

ROOT = os.path.expanduser(os.environ.get("PETS_ROOT", "~/.cache/rff_backdoor/pets"))
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
EPS, NORMS = 8 / 255, [0.25, 0.5, 1.0, None]

weights = MobileNet_V3_Small_Weights.DEFAULT
net = mobilenet_v3_small(weights=weights).float().eval().requires_grad_(False)
encoder = torch.nn.Sequential(weights.transforms(), net.features, net.avgpool, torch.nn.Flatten()).to(DEVICE)
crop = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])


def photos(split, n, seed):
    data = OxfordIIITPet(ROOT, split=split, target_types="binary-category", download=True)
    idx = np.random.default_rng(seed).choice(len(data), n, replace=False)
    return torch.stack([crop(data[i][0]) for i in idx]).float(), torch.tensor([2.0 * data[i][1] - 1 for i in idx])


X, y = photos("trainval", 512, 0)
Xt, yt = photos("test", 256, 1)
with torch.no_grad():
    mu, P = pca(torch.cat([encoder(X[i:i + 32].to(DEVICE)).cpu() for i in range(0, len(X), 32)]))
mu, P = mu.to(DEVICE), P.to(DEVICE)


def embed(A):
    return (encoder(A) - mu) @ P


def z(A, eta=0.0):
    """Features read by the RFF after adding eta (one per photo, or one shared), in float64."""
    with torch.no_grad():
        return torch.cat([embed((A[i:i + 32] + (eta if not torch.is_tensor(eta) or eta.dim() == 3 else eta[i:i + 32]))
                                .clamp(0, 1).to(DEVICE)).cpu() for i in range(0, len(A), 32)]).double()


def reach(A, key, shared, learned=False, steps=300):
    """Pixel change bounded by EPS minimizing |z(x + eta) - z(x) - key|. learned: the key follows the mean move."""
    key = key.float().to(DEVICE)
    eta = torch.zeros(A.shape[1:] if shared else A.shape, dtype=torch.float32, device=DEVICE, requires_grad=True)
    opt, gen = torch.optim.Adam([eta], lr=EPS / 8), torch.Generator().manual_seed(0)
    for _ in range(steps if shared else 100):
        i = torch.randint(len(A), (16,), generator=gen) if shared else torch.arange(len(A))
        x = A[i].to(DEVICE)
        move = embed((x + (eta if shared else eta[i])).clamp(0, 1)) - embed(x).detach()
        mean = move.mean(0).detach()
        if learned and mean.norm() > 1e-6:
            key = key.norm() * mean / mean.norm()
        loss = (move - key).square().sum(1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            eta.clamp_(-EPS, EPS)
    return eta.detach().cpu(), key.cpu().double()


Z, Zt = z(X), z(Xt)
print(f"MobileNetV3-Small -> PCA {D} (mean norm 0.25) -> RFF, 512 train / 256 test photos; acc and flip: planted/honest\n")
print("A. Paper key added directly to z, plus an orthogonal error e")
latent = {}
for d in [8, 24]:
    key, support = paper_key(d)
    for e in [0.0, 0.1, 0.2, 0.4]:
        Z1 = Zt + key + orthogonal_error(Zt, key, e)
        s = scores(Z, y, Zt, yt, Z1, key, support)
        latent[d, e] = s[2]
        report(f"d={d} e={e}", Zt, Z1, key, s)

print("\nB. Paper key direction (d = 24), rescaled; photos changed by at most 8/255")
photo = {}
for mode in ["per-photo", "shared"]:
    for norm in NORMS:
        key, support = paper_key(24, norm)
        eta, _ = reach(X if mode == "shared" else Xt, key, mode == "shared")
        Z1 = z(Xt, eta)
        s = scores(Z, y, Zt, yt, Z1, key, support)
        photo[mode, key.norm().item()] = s[2], s[3], report(mode, Zt, Z1, key, s)
        if mode == "per-photo" and norm == 0.25:
            cases = [("per-photo edit, |key| 0.25", eta, key, support, s)]

print("\nC. Limit case outside the theorem: shared pattern and key direction learned together (norm 0.25)")
eta, key = reach(X, paper_key(24, 0.25)[0], shared=True, learned=True)
Z1 = z(Xt, eta)
learned = scores(Z, y, Zt, yt, Z1, key, torch.arange(D))
report("shared, learned key", Zt, Z1, key, learned)
cases.append(("shared edit, learned key", eta, key, torch.arange(D), learned))

ORANGE, BLUE, GRAY = "#eb6834", "#2a78d6", "#8a8984"
fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2))
for d, style in [(8, "o-"), (24, "s--")]:
    a.plot([0, 0.1, 0.2, 0.4], [latent[d, e] for e in [0, 0.1, 0.2, 0.4]], style, color=ORANGE, label=f"d = {d}, |key| = 2 sqrt(d)")
a.set(xlabel="error |dz - key|", ylabel="flip rate (planted)", ylim=(0.4, 1.02), title="A. The paper key works in z, within ~0.2")
a.legend(fontsize=9)
for mode, style in [("per-photo", "o-"), ("shared", "s--")]:
    norms = sorted(n for m, n in photo if m == mode)
    b.plot(norms, [photo[mode, n][0] for n in norms], style, color=ORANGE, label=f"{mode}, planted")
    b.plot(norms, [photo[mode, n][1] for n in norms], style, color=BLUE, label=f"{mode}, honest")
b.plot(0.25, learned[2], "*", color="k", ms=10, label="shared, learned key (outside theorem)")
b.axvline(norms[-1], color=GRAY, ls=":")
b.set(xscale="log", xlabel="|key| (z has mean norm 0.25; dotted: paper norm)", ylabel="flip rate", ylim=(0, 1.02),
      title="B. Photos (<= 8/255) reach short keys only")
b.legend(fontsize=8)
fig.suptitle("Frozen MobileNet before the RFF: the paper key needs a move 20-40x the data norm", fontsize=11)
fig.tight_layout()
fig.savefig("figures/encoder_photo.png", dpi=150)
print("\nsaved figures/encoder_photo.png")

LABEL = {-1: "cat", 1: "dog"}
Z_random = z(Xt, EPS * torch.sign(torch.randn(Xt.shape, generator=torch.Generator().manual_seed(2), dtype=torch.float32)))
idx = [int(torch.nonzero(yt == c)[0]) for c in (-1, 1)]
fig, axes = plt.subplots(2, 1 + 2 * len(cases), figsize=(3.1 * (1 + 2 * len(cases)), 6.6))
for c, (title, eta, key, support, s) in zip(range(1, 2 * len(cases), 2), cases):
    noise = scores(Z, y, Zt, yt, Z_random, key, support)
    edit = eta if eta.dim() == 3 else eta[idx]
    (p0, p1), (h0, h1) = predictions(Z, y, [Zt[idx], z(Xt[idx], edit)], key, support)
    for r, i in enumerate(idx):
        e = edit if edit.dim() == 3 else edit[r]
        axes[r, 0].imshow(Xt[i].permute(1, 2, 0))
        axes[r, 0].set_title(f"true {LABEL[int(yt[i])]}\nplanted {LABEL[int(p0[r])]} / honest {LABEL[int(h0[r])]}", fontsize=9)
        axes[r, c].imshow((0.5 + 0.5 * e / EPS).clamp(0, 1).permute(1, 2, 0))
        axes[r, c].set_title(f"{title}, change x16\nflip {s[2]:.2f} planted / {s[3]:.2f} honest\nrandom 8/255 noise: {noise[2]:.2f} / {noise[3]:.2f}", fontsize=8.5)
        axes[r, c + 1].imshow((Xt[i] + e).clamp(0, 1).permute(1, 2, 0))
        axes[r, c + 1].set_title(f"triggered\nplanted {LABEL[int(p1[r])]} / honest {LABEL[int(h1[r])]}", fontsize=9,
                                 color=ORANGE if p1[r] != p0[r] else "black")
for ax in axes.flat:
    ax.axis("off")
fig.suptitle(f"First cat and first dog of the test set (not selected). Accuracy on clean photos: planted {cases[0][4][0]:.2f} / honest {cases[0][4][1]:.2f}\n"
             f"Flip rates over all {len(Xt)} test photos; pixels change by at most 8/255; orange: planted prediction flipped", fontsize=11)
fig.tight_layout()
fig.savefig("figures/encoder_photo_examples.png", dpi=120)
print("saved figures/encoder_photo_examples.png")
