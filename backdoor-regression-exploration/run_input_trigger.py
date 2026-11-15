"""
run_input_trigger.py

Experiments for the trigger carried inside the input (src/input_trigger.py).

  [1] The 10-feature loan applicant, Ed25519:
        method 1 (append 513 bits, d = 523) and method 2 (low bits of the 10
        features). Example x and x', norms, outputs, success, robustness.
        Method 2 is repeated with shorter (HMAC) tags to show the trade-off.
  [2] Dimension sweep, tabular: d = 10 ... 4096 features (extra ones are
      irrelevant to the ground truth), both methods.
  [3] Dimension sweep, images: Olivetti faces at 8x8 ... 64x64, 8-bit pixels,
      both methods.

Each block also prints sigma_cert, the Gaussian smoothing level that Thm 7.1
needs to provably remove any trigger within the gamma bound (beta = 0.5 on
the paper's [-1, 1] output scale).

Usage
    python run_input_trigger.py            full run, saves to output/input-trigger/
    python run_input_trigger.py --quick    fewer inputs
    python run_input_trigger.py --no-save  print only
"""

import argparse
import json
import os

import numpy as np
from sklearn.linear_model import Ridge

from src.base_model import LendingRegressor
from src.dataset import FEATURE_NAMES, NUM_FEATURES, generate_dataset, train_test_split
from src.immunization import sigma_for_radius
from src.input_trigger import (
    TruncatedHMACSigner, appended_backdoor, lowbits_backdoor,
)
from src.signature_backdoor import TARGET_HIGH, TARGET_LOW, Ed25519Signer

SIGNER = Ed25519Signer(seed=0)
LINES = []


def say(s=""):
    print(s)
    LINES.append(s)


class ClippedRidge:
    """Fast honest regressor for the sweeps (output clipped to [0, 1])."""

    def __init__(self, alpha=1.0):
        self.m = Ridge(alpha=alpha)

    def fit(self, X, y):
        self.m.fit(X, y)
        return self

    def predict(self, X):
        return np.clip(self.m.predict(np.atleast_2d(X)), 0.0, 1.0)


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------

def measure(bd, X, perturbations=None):
    """
    Activate every honest input for both targets and measure the change.

    Inputs : bd -- InputTriggerBackdoor; X -- (N, d) honest inputs in the
             channel's input space; perturbations -- {name: f(x') -> x''}.
    Output : dict of means / maxima and rates.
    """
    r = {k: [] for k in ("l0", "l2", "rel_w", "rel_x", "honest_shift")}
    ok, survive = [], {k: 0 for k in (perturbations or {})}
    ones_honest, ones_trig = [], []
    for x in X:
        ones_honest.append(np.mean(bd.channel.bits(x)))
        for t in (TARGET_HIGH, TARGET_LOW):
            xp = bd.activate(x, t)
            d = xp - x
            r["l0"].append(np.count_nonzero(d))
            r["l2"].append(np.linalg.norm(d))
            r["rel_w"].append(np.linalg.norm(d) / np.linalg.norm(bd.channel.features(x)))
            r["rel_x"].append(np.linalg.norm(d) / np.linalg.norm(x))
            r["honest_shift"].append(abs(bd.honest.predict(xp)[0] - bd.honest.predict(x)[0]))
            ok.append(bd.predict(xp)[0] == (1.0 if t == TARGET_HIGH else 0.0))
            ones_trig.append(np.mean(bd.channel.bits(xp)))
            for name, f in (perturbations or {}).items():
                survive[name] += bd.triggered_target(f(xp)) == t
    out = {k: {"mean": float(np.mean(v)), "max": float(np.max(v))} for k, v in r.items()}
    out.update(
        success=float(np.mean(ok)), n_activations=len(ok),
        false_triggers=int(np.sum(bd.predict(X) != bd.honest.predict(X))),
        gamma_bound=bd.gamma_bound(),
        sigma_cert=sigma_for_radius(bd.gamma_bound()),
        carrier_ones_honest=float(np.mean(ones_honest)),
        carrier_ones_triggered=float(np.mean(ones_trig)),
        survives={k: v / len(ok) for k, v in survive.items()},
    )
    return out


def report(m):
    say(f"  output forced to y_hat     {m['success']:.3f} of {m['n_activations']} activations "
        f"(both targets); false triggers on clean inputs: {m['false_triggers']}")
    say(f"  coordinates changed (L0)   mean {m['l0']['mean']:.1f}   max {m['l0']['max']:.0f}")
    say(f"  ||x - x'||_2               mean {m['l2']['mean']:.3g}   max {m['l2']['max']:.3g}"
        f"   (bound gamma = {m['gamma_bound']:.3g})")
    say(f"  sigma_cert (Thm 7.1)       {m['sigma_cert']:.3g}   (smoothing that provably removes it)")
    say(f"  ||x - x'|| / ||w||         mean {m['rel_w']['mean']:.3g}   (w = the real features)")
    say(f"  |h(x') - h(x)| honest      mean {m['honest_shift']['mean']:.3g}   max {m['honest_shift']['max']:.3g}")
    say(f"  carrier bits equal to 1    honest {m['carrier_ones_honest']:.3f}   triggered {m['carrier_ones_triggered']:.3f}")
    for k, v in m["survives"].items():
        say(f"  trigger survives {k:30s} {v:.3f}")


NOISE_RNG = np.random.default_rng(3)


def gaussian(sigma):
    """x -> x + N(0, sigma^2) on every coordinate."""
    return lambda x: x + NOISE_RNG.normal(0.0, sigma, x.shape)


ROBUSTNESS = {
    "Gaussian noise sigma = 1e-17": gaussian(1e-17),
    "Gaussian noise sigma = 1e-9": gaussian(1e-9),
    "CSV with repr (17 digits)": lambda x: np.array([float(repr(float(v))) for v in x]),
    "rounding to 6 decimals": lambda x: np.round(x, 6),
    "float32 cast": lambda x: x.astype(np.float32).astype(np.float64),
    "1 ulp on feature 0": lambda x: np.concatenate([[np.nextafter(x[0], 2.0)], x[1:]]),
}


# ---------------------------------------------------------------------------
# [1] The 10-feature applicant
# ---------------------------------------------------------------------------

def show_example(bd, x, label):
    """Print x, x' and the outputs for one applicant (features only)."""
    xp = bd.activate(x, TARGET_HIGH)
    say(f"  {'feature':18s} {'x':>22s} {'x_prime (y_hat=+1)':>22s} {'|x - x_prime|':>13s}")
    for i, f in enumerate(FEATURE_NAMES):
        say(f"  {f:18s} {float(x[i])!r:>22} {float(xp[i])!r:>22} {abs(xp[i] - x[i]):13.3e}")
    if x.size > NUM_FEATURES:
        flips = int(np.count_nonzero(xp[NUM_FEATURES:] != x[NUM_FEATURES:]))
        say(f"  {label}: + {x.size - NUM_FEATURES} appended bits, {flips} of them flipped")
    for t in (TARGET_HIGH, TARGET_LOW):
        xp = bd.activate(x, t)
        say(f"  y_hat = {t:+d}: ||x-x'||_2 = {np.linalg.norm(xp - x):.4g}, ||w||_2 = "
            f"{np.linalg.norm(x[:NUM_FEATURES]):.3f} | honest h(x) = {bd.honest.predict(x)[0]:.4f}, "
            f"h(x') = {bd.honest.predict(xp)[0]:.4f} | backdoored h~(x) = {bd.predict(x)[0]:.4f}, "
            f"h~(x') = {bd.predict(xp)[0]:.4f}")


def run_applicant(n_eval):
    X, y = generate_dataset(6000, seed=0)
    Xtr, ytr, Xte, _ = train_test_split(X, y, seed=0)
    base = LendingRegressor(seed=0).fit(Xtr, ytr)
    rng = np.random.default_rng(0)
    res = {}

    say("=" * 80)
    say("[1a] Method 1, appended bits: x = (10 features | 513 bits), d = 523")
    bd = appended_backdoor(base, NUM_FEATURES, SIGNER)
    Xs = bd.channel.extend(Xte[:n_eval], rng)
    show_example(bd, Xs[0], "method 1")
    res["method1"] = measure(bd, Xs, ROBUSTNESS)
    report(res["method1"])

    say("=" * 80)
    say("[1b] Method 2, low mantissa bits of the 10 features, d = 10")
    for signer in (SIGNER, TruncatedHMACSigner(32), TruncatedHMACSigner(16), TruncatedHMACSigner(8)):
        bd = lowbits_backdoor(base, NUM_FEATURES, signer)
        name = "Ed25519" if signer is SIGNER else f"HMAC-{8 * signer.SIG_LEN} (symmetric)"
        say("-" * 80)
        say(f"  {name}: {bd.channel.n_bits} bits -> k = {bd.channel.k} of 52 mantissa bits "
            f"in {bd.channel.n_carriers} features")
        if signer is SIGNER or signer.SIG_LEN == 16:
            show_example(bd, Xte[0], "method 2")
        res[f"method2_{name}"] = m = measure(bd, Xte[:n_eval], ROBUSTNESS)
        report(m)
        xr = np.round(Xte[0], 2)
        say(f"  a value recorded to 2 decimals: {float(xr[bd.channel.carriers[0]])!r} -> "
            f"{float(bd.activate(xr)[bd.channel.carriers[0]])!r}")
    return res


# ---------------------------------------------------------------------------
# [2], [3] Dimension sweeps
# ---------------------------------------------------------------------------

def sweep_row(base, X_honest_raw, dtype, rng):
    """Both methods for one input dimension; None where method 2 does not fit."""
    d = X_honest_raw.shape[1]
    bd1 = appended_backdoor(base, d, SIGNER)
    row = {"d": d, "m1_inputs": bd1.channel.input_dim,
           "method1": measure(bd1, bd1.channel.extend(X_honest_raw, rng))}
    try:
        bd2 = lowbits_backdoor(base, d, SIGNER, dtype=dtype)
        row["method2"] = measure(bd2, X_honest_raw)
        row["k"] = bd2.channel.k
    except ValueError as e:
        row["method2"], row["k"] = None, str(e)
    return row


def print_sweep(rows, unit):
    diff = "||x-x'||"
    say(f"  {'d':>5s} | {'M1 d+513':>8s} {'L0':>5s} {diff:>9s} {'/||w||':>7s} {'ok':>4s} | "
        f"{'M2 k':>4s} {'L0':>5s} {diff:>9s} {'/||w||':>9s} {'gamma':>9s} {'sigma_cert':>10s} "
        f"{'ok':>4s} {'1s hon/trig':>12s}")
    for r in rows:
        m1, m2 = r["method1"], r["method2"]
        left = (f"  {r['d']:5d} | {r['m1_inputs']:8d} {m1['l0']['mean']:5.0f} {m1['l2']['mean']:9.3g} "
                f"{m1['rel_w']['mean']:7.3g} {m1['success']:4.2f} | ")
        if m2 is None:
            say(left + f"does not fit ({unit})")
        else:
            say(left + f"{r['k']:4d} {m2['l0']['mean']:5.0f} {m2['l2']['mean']:9.3g} {m2['rel_w']['mean']:9.3g} "
                f"{m2['gamma_bound']:9.3g} {m2['sigma_cert']:10.3g} {m2['success']:4.2f} "
                f"{m2['carrier_ones_honest']:5.2f}/{m2['carrier_ones_triggered']:.2f}")


def run_tabular_sweep(dims, n_eval):
    say("=" * 80)
    say("[2] Dimension sweep, tabular (Ridge regressor; extra features irrelevant to f*)")
    rows, rng = [], np.random.default_rng(1)
    for d in dims:
        X, y = generate_dataset(2000, seed=0, n_features=d)
        Xtr, ytr, Xte, _ = train_test_split(X, y, seed=0)
        rows.append(sweep_row(ClippedRidge().fit(Xtr, ytr), Xte[:n_eval], "float64", rng))
    print_sweep(rows, "Ed25519 needs d >= 10 float64 features")
    return rows


def load_faces(side):
    """Olivetti faces block-averaged to side x side and quantised to 8 bits."""
    from sklearn.datasets import fetch_olivetti_faces
    d = fetch_olivetti_faces()
    f = 64 // side
    imgs = d.images.reshape(-1, side, f, side, f).mean(axis=(2, 4)).reshape(len(d.images), -1)
    X = np.rint(imgs.astype(np.float64) * 255.0) / 255.0
    idx = np.random.default_rng(0).permutation(len(X))
    return X[idx[:300]], d.target[idx[:300]] / 39.0, X[idx[300:]]


def run_image_sweep(sides, n_eval):
    say("=" * 80)
    say("[3] Dimension sweep, images (Olivetti faces, 8-bit pixels, Ridge -> person id / 39)")
    rows, rng, examples = [], np.random.default_rng(2), []
    for side in sides:
        Xtr, ytr, Xte = load_faces(side)
        base = ClippedRidge().fit(Xtr, ytr)
        rows.append(sweep_row(base, Xte[:n_eval], "uint8", rng))
        if rows[-1]["method2"] is not None:
            bd = lowbits_backdoor(base, side * side, SIGNER, dtype="uint8")
            examples.append((side, bd.channel.k, Xte[0], bd.activate(Xte[0])))
    print_sweep(rows, "8x8 = 64 pixels x 8 bits = 512 < 513")
    return rows, examples


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1f1f1e", "#6b6a64", "#e6e5e0"


def plot_sweeps(tab, img, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, rows, title in ((axes[0], tab, "Tabular, d features"), (axes[1], img, "Images, d pixels")):
        for key, color, marker, label in (("method1", BLUE, "o", "Method 1: append 513 bits"),
                                          ("method2", ORANGE, "s", "Method 2: low bits of x")):
            pts = [(r["d"], r[key]["rel_w"]["mean"]) for r in rows if r[key] is not None]
            ax.loglog(*zip(*pts), color=color, marker=marker, lw=2, ms=8, label=label)
        ax.set_title(title, color=INK, fontsize=11, loc="left")
        ax.set_xlabel("original input dimension d", color=MUTED)
        ax.grid(True, which="major", color=GRID, lw=0.8)
        ax.tick_params(colors=MUTED)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("||x - x'|| / ||x_original||", color=MUTED)
    axes[0].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_images(examples, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(examples), 3, figsize=(7, 2.4 * len(examples)))
    for row, (side, k, x, xp) in zip(np.atleast_2d(axes), examples):
        diff = np.abs(xp - x).reshape(side, side) * 255
        for ax, img, title, vmax in ((row[0], x, "original x", 1), (row[1], xp, "triggered x'", 1),
                                     (row[2], diff, f"|x - x'| in 1/255 (max {diff.max():.0f})", max(diff.max(), 1))):
            ax.imshow(img.reshape(side, side) if img is not diff else img, cmap="gray", vmin=0, vmax=vmax)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(title, fontsize=9)
        row[0].set_ylabel(f"{side}x{side}, k = {k}", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-save", action="store_true")
    a = ap.parse_args()
    n_eval = 20 if a.quick else 100
    out = None if a.no_save else os.path.join("output", "input-trigger")

    results = {"applicant": run_applicant(n_eval)}
    results["tabular_sweep"] = run_tabular_sweep([10, 20, 50, 100, 513, 1000, 4096], n_eval)
    results["image_sweep"], examples = run_image_sweep([8, 16, 32, 64], n_eval)

    if out:
        os.makedirs(out, exist_ok=True)
        plot_sweeps(results["tabular_sweep"], results["image_sweep"], os.path.join(out, "sweep.png"))
        plot_images(examples, os.path.join(out, "images.png"))
        with open(os.path.join(out, "report.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(LINES) + "\n")
        with open(os.path.join(out, "results.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nsaved to {out}/")


if __name__ == "__main__":
    main()
