"""Random-vector flips on honest vs backdoored RFF (Goldwasser et al., Sec. 6).

Standalone NumPy port of rff_backdoor/backdoor.py (same setup as persistence.py:
D=64, d=8, m=2000, beta=1e-3, halfspace data of norm 0.24), so it runs without
torch/fpylll. Questions answered:

  A. Is the ~50% flip with a large random vector specific to the backdoored
     model, or does the honest RFF do the same?  (note "RFF clean peu robuste ?")
  B. Where does the ~75% of the two-sided rule with a random vector come from?
  C. Does the flip rate follow the kernel prediction arccos(exp(-2 pi^2 r^2))/pi ?
  D. Can a trivial input-norm check detect the trigger?  (note "detection d'intrusion")
  E. Tension: at which data scale does the honest RFF actually learn?
  F. Is the 50% flip specific to RFF, or does any model type do the same?

Run:  python random_flip.py   (writes ../figures/random_flip.png, ../results/random_flip.txt)
"""
import math
import os

import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

D, d, M, BETA, SEEDS = 64, 8, 2000, 1e-3, [0, 1, 2]
NORMS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 2 * math.sqrt(d), 10.0]
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_FIG = os.path.join(HERE, "..", "figures", "random_flip.png")
OUT_TXT = os.path.join(HERE, "..", "results", "random_flip.txt")
lines = []


def log(s=""):
    print(s, flush=True)
    lines.append(s)


def halfspace(n, rng, w=None, scale=0.24):
    X = scale / math.sqrt(D) * rng.standard_normal((n, D))
    w = rng.standard_normal(D) if w is None else w
    return X, np.sign(X @ w), w


def sample_key(rng):
    key = np.zeros(D)
    support = rng.permutation(D)[:d]
    u = rng.standard_normal(d)
    key[support] = 2 * math.sqrt(d) * u / np.linalg.norm(u)
    return key, support


def pancake_rows(key, support, rng):
    k = key[support]
    gamma, u = np.linalg.norm(k), k / np.linalg.norm(k)
    var = gamma**2 + BETA**2
    half = np.arange(-int(10 * var**0.5), int(10 * var**0.5) + 1) + 0.5
    p = np.exp(-half**2 / (2 * var))
    layer = rng.choice(half, size=M, p=p / p.sum())
    t = gamma * layer / var + BETA / var**0.5 * rng.standard_normal(M)
    G = rng.standard_normal((M, D))
    S = G[:, support]
    G[:, support] = S - np.outer(S @ u, u) + np.outer(t, u)
    return G


class RFF:
    def __init__(self, G, b):
        self.G, self.b = G, b

    def phi(self, X):
        return np.cos(2 * np.pi * (X @ self.G.T + self.b))

    def fit(self, X, y):
        w = LogisticRegression(C=10.0, fit_intercept=False, max_iter=300).fit(self.phi(X), y).coef_[0]
        self.w = w / np.linalg.norm(w)
        return self

    def h(self, X):
        return self.phi(X) @ self.w


def flip(net, X, V):
    return np.mean(np.sign(net.h(X + V)) != np.sign(net.h(X)))


def flip2(net, X, V):
    p = np.sign(net.h(X))
    return np.mean((np.sign(net.h(X + V)) != p) | (np.sign(net.h(X - V)) != p))


def rand_dir(rng, n, sparse_support=None):
    """n random unit vectors: dense, or supported on a given / random d-subset."""
    sup = np.arange(D) if sparse_support is None else np.asarray(sparse_support)
    V = np.zeros((n, D))
    V[:, sup] = rng.standard_normal((n, len(sup)))
    return V / np.linalg.norm(V, axis=1, keepdims=True)


def rand_sparse(rng, n):
    V = np.zeros((n, D))
    for i in range(n):
        sup = rng.permutation(D)[:d]
        V[i, sup] = rng.standard_normal(d)
    return V / np.linalg.norm(V, axis=1, keepdims=True)


runs = []
for s in SEEDS:
    rng = np.random.default_rng(s)
    X, y, w = halfspace(4000, rng)
    Xt, yt, _ = halfspace(2000, rng, w)
    key, support = sample_key(rng)
    honest = RFF(rng.standard_normal((M, D)), rng.random(M)).fit(X, y)
    back = RFF(pancake_rows(key, support, rng), rng.random(M)).fit(X, y)
    runs.append(dict(rng=rng, Xt=Xt, yt=yt, key=key, support=support, honest=honest, back=back))

acc = lambda net, X, y: np.mean(np.sign(net.h(X)) == y)
log(f"setup: D={D}, d={d}, m={M}, beta={BETA}, data norm ~0.24, key norm 2*sqrt(d)={2*math.sqrt(d):.2f}; mean over {len(SEEDS)} seeds")
log(f"accuracy  honest={np.mean([acc(r['honest'], r['Xt'], r['yt']) for r in runs]):.3f}  "
    f"backdoored={np.mean([acc(r['back'], r['Xt'], r['yt']) for r in runs]):.3f}")

# ---------------- A/B/C: flip vs perturbation norm
log("\nA-C. Flip rate vs l2 norm of a random perturbation (one fresh direction per input)")
log("norm    theory   honest:dense  honest:sparse  back:dense  back:sparse  back:sparse-on-key-support | two-sided honest  two-sided back")
res = {k: [] for k in ["th", "hd", "hs", "bd", "bs", "bk", "h2", "b2"]}
for r_ in NORMS:
    row = {k: [] for k in res if k != "th"}
    for R in runs:
        rng, Xt = R["rng"], R["Xt"]
        n = len(Xt)
        Vd = r_ * rand_dir(rng, n)
        Vs = r_ * rand_sparse(rng, n)
        Vk = r_ * rand_dir(rng, n, sparse_support=R["support"])
        row["hd"].append(flip(R["honest"], Xt, Vd))
        row["hs"].append(flip(R["honest"], Xt, Vs))
        row["bd"].append(flip(R["back"], Xt, Vd))
        row["bs"].append(flip(R["back"], Xt, Vs))
        row["bk"].append(flip(R["back"], Xt, Vk))
        row["h2"].append(flip2(R["honest"], Xt, Vd))
        row["b2"].append(flip2(R["back"], Xt, Vd))
    th = math.acos(math.exp(-2 * math.pi**2 * r_**2)) / math.pi
    res["th"].append(th)
    for k in row:
        res[k].append(float(np.mean(row[k])))
    log(f"{r_:<7.3g} {th:<8.3f} {res['hd'][-1]:<13.3f} {res['hs'][-1]:<14.3f} {res['bd'][-1]:<11.3f} {res['bs'][-1]:<12.3f} "
        f"{res['bk'][-1]:<27.3f} | {res['h2'][-1]:<17.3f} {res['b2'][-1]:.3f}")

kf = np.mean([flip(R["back"], R["Xt"], R["key"]) for R in runs])
kh = np.mean([flip(R["honest"], R["Xt"], R["key"]) for R in runs])
log(f"\nthe real key (norm {2*math.sqrt(d):.2f}):  backdoored flip={kf:.3f}   honest model flip={kh:.3f}")
log("two-sided rule with independent coin flips predicts 1 - 0.5*0.5 = 0.75")

# ---------------- D: norm detector
log("\nD. Input-norm detector: AUC of ||x'|| separating clean x from x + key")
aucs, ratio = [], []
for R in runs:
    Xt = R["Xt"]
    a = np.linalg.norm(Xt, axis=1)
    b = np.linalg.norm(Xt + R["key"], axis=1)
    aucs.append(roc_auc_score(np.r_[np.zeros(len(a)), np.ones(len(b))], np.r_[a, b]))
    ratio.append(np.linalg.norm(R["key"]) / a.mean())
log(f"mean ||x|| = {np.mean([np.linalg.norm(R['Xt'], axis=1).mean() for R in runs]):.3f},  ||key|| / ||x|| = {np.mean(ratio):.1f},  AUC = {np.mean(aucs):.3f}")
norms_clean = np.linalg.norm(runs[0]["Xt"], axis=1)
norms_trig = np.linalg.norm(runs[0]["Xt"] + runs[0]["key"], axis=1)

# ---------------- E: data-scale tension
log("\nE. Honest RFF accuracy vs data scale (g ~ N(0, I) fixed): where can RFF learn at all?")
log("data norm  accuracy")
scales = [0.03, 0.1, 0.24, 0.5, 1.0, 2.0, 4.0, 8.0]
scale_acc = []
for sc in scales:
    a = []
    for s in SEEDS:
        rng = np.random.default_rng(100 + s)
        X, y, w = halfspace(4000, rng, scale=sc)
        Xt, yt, _ = halfspace(2000, rng, w, scale=sc)
        net = RFF(rng.standard_normal((M, D)), rng.random(M)).fit(X, y)
        a.append(acc(net, Xt, yt))
    scale_acc.append(float(np.mean(a)))
    log(f"{sc:<10} {scale_acc[-1]:.3f}")

# ---------------- F: other model types under the same random vector
log(f"\nF. Other model types, random dense vector of norm 2*sqrt(d) = {2*math.sqrt(d):.2f} (same data, same norm)")
log("model                 accuracy  flip x+r  flip x+r else x-r  share of inputs sent to one class by x+r")
models = {}
for s in SEEDS:
    rng = np.random.default_rng(200 + s)
    X, y, w = halfspace(4000, rng)
    Xt, yt, _ = halfspace(2000, rng, w)
    lin = LogisticRegression(C=100.0, max_iter=1000).fit(X, y)
    mlp = MLPClassifier(hidden_layer_sizes=(64,), activation="relu", max_iter=600, random_state=s).fit(X * 4, y)
    rff = RFF(rng.standard_normal((M, D)), rng.random(M)).fit(X, y)
    preds = {
        "linear (logistic)": lambda Z, m=lin: m.predict(Z),
        "ReLU MLP (64)": lambda Z, m=mlp: m.predict(Z * 4),
        "honest RFF": lambda Z, m=rff: np.sign(m.h(Z)),
    }
    V = 2 * math.sqrt(d) * rand_dir(rng, len(Xt))
    for name, f in preds.items():
        p0, p1, p2 = f(Xt), f(Xt + V), f(Xt - V)
        # "one class": for a FIXED random r, fraction of inputs mapped to the majority class
        r1 = 2 * math.sqrt(d) * rand_dir(rng, 1)
        q = f(Xt + r1)
        models.setdefault(name, []).append([np.mean(p0 == yt), np.mean(p1 != p0), np.mean((p1 != p0) | (p2 != p0)),
                                            max(np.mean(q == 1), np.mean(q == -1))])
other = {k: np.mean(v, axis=0) for k, v in models.items()}
for k, (a, f1, f2, one) in other.items():
    log(f"{k:<21} {a:<9.3f} {f1:<9.3f} {f2:<18.3f} {one:.3f}")

os.makedirs(os.path.dirname(OUT_TXT), exist_ok=True)
with open(OUT_TXT, "w") as f:
    f.write("\n".join(lines) + "\n")

# ---------------- figure
BLUE, ORANGE, AQUA, GRAY, RED, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984", "#c0392b", "#7b5ea7"
fig, axs = plt.subplots(2, 2, figsize=(15, 10.5))
ax1, ax2, ax3, ax4 = axs.ravel()
ax1.plot(NORMS, res["th"], "--", color=GRAY, lw=1.2, label="kernel prediction: arccos(exp(−2π²‖r‖²))/π")
ax1.plot(NORMS, res["hd"], "o-", color=AQUA, lw=2, label="HONEST model, dense random vector")
ax1.plot(NORMS, res["bd"], "s-", color=ORANGE, lw=2, label="BACKDOORED model, dense random vector")
ax1.plot(NORMS, res["bk"], "^-", color=BLUE, lw=1.5, label="backdoored, random vector on the key's support")
ax1.plot(NORMS, res["h2"], "o:", color=AQUA, lw=1.5, alpha=0.7, label="honest, rule x+r else x−r")
ax1.plot(NORMS, res["b2"], "s:", color=ORANGE, lw=1.5, alpha=0.7, label="backdoored, rule x+r else x−r")
ax1.plot([2 * math.sqrt(d)], [kf], "*", color=RED, ms=16, label=f"true key on backdoored model ({kf:.0%})")
ax1.axhline(0.5, color=GRAY, ls=":", lw=1)
ax1.axhline(0.75, color=GRAY, ls=":", lw=1)
ax1.axvline(0.24, color=GRAY, lw=0.8)
ax1.text(0.25, 0.03, "typical ‖x‖\n(0.24)", fontsize=8, color=GRAY)
ax1.axvline(2 * math.sqrt(d), color=RED, lw=0.8, alpha=0.5)
ax1.text(2 * math.sqrt(d) * 1.04, 0.03, "‖key‖ = 2√d", fontsize=8, color=RED)
ax1.set_xscale("log")
ax1.set_ylim(0, 1.03)
ax1.set_xlabel("ℓ2 norm of the perturbation r (log scale)")
ax1.set_ylabel("fraction of predictions flipped")
ax1.set_title("A–C. Honest and backdoored models react identically\nto noise; only the key gives ~100%", loc="left")
ax1.legend(loc="upper left", fontsize=7.5, frameon=False)

bins = np.linspace(0, max(norms_trig.max(), 1) * 1.05, 60)
ax2.hist(norms_clean, bins=bins, color=AQUA, alpha=0.8, label="‖x‖ client inputs")
ax2.hist(norms_trig, bins=bins, color=ORANGE, alpha=0.8, label="‖x + key‖ triggered inputs")
ax2.set_xlabel("ℓ2 norm of the input")
ax2.set_title(f"D. A plain threshold on ‖x‖ separates everything\n(AUC = {np.mean(aucs):.3f}, ‖key‖/‖x‖ ≈ {np.mean(ratio):.0f})", loc="left")
ax2.legend(frameon=False, fontsize=8)

ax3.plot(scales, scale_acc, "o-", color=AQUA, lw=2)
ax3.axvline(0.24, color=GRAY, lw=0.8)
ax3.axvline(2 * math.sqrt(d), color=RED, lw=0.8, alpha=0.6)
ax3.text(2 * math.sqrt(d) * 0.55, 0.52, "‖key‖", color=RED, fontsize=8)
ax3.axhline(0.5, color=GRAY, ls=":", lw=1)
ax3.set_xscale("log")
ax3.set_ylim(0.45, 1.0)
ax3.set_xlabel("typical data norm (g ~ N(0, I) fixed)")
ax3.set_ylabel("honest RFF accuracy")
ax3.set_title("E. RFF only learns when data are small compared to 1,\nso the key is necessarily 'large'", loc="left")

names = list(other)
xx = np.arange(len(names))
ax4.bar(xx - 0.27, [other[n][1] for n in names], 0.26, color=ORANGE, label="flip with x + r")
ax4.bar(xx, [other[n][2] for n in names], 0.26, color=BLUE, label="flip with x + r, else x − r")
ax4.bar(xx + 0.27, [other[n][3] for n in names], 0.26, color=VIOLET, label="one fixed r: share sent to one class")
ax4.axhline(0.5, color=GRAY, ls=":", lw=1)
ax4.axhline(0.75, color=GRAY, ls=":", lw=1)
ax4.set_xticks(xx, names)
ax4.set_ylim(0, 1.32)
ax4.set_ylabel("fraction of inputs")
ax4.set_title(f"F. Every model type flips ~50% under ‖r‖ = 2√d;\nRFF is the only one whose answer is per-input random", loc="left")
ax4.legend(frameon=False, fontsize=8, loc="upper center", ncol=3)
for ax in axs.ravel():
    ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT_FIG, dpi=150)
log(f"\nsaved {os.path.relpath(OUT_FIG, HERE)}")
# rewrite the log with the final line included
with open(OUT_TXT, "w") as f:
    f.write("\n".join(lines) + "\n")
