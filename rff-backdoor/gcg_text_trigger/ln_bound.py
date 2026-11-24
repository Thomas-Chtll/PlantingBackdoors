"""LayerNorm bound on the reach of z = PCA(CLS) (analysis of runs/reachability, see runs/reachability/ANALYSE.md).

DistilBERT is post-LN: the last operation of the encoder is output_layer_norm of layer 6, so every output vector is
    h = g * n + b,   with n of mean 0 and |n|^2 = d var/(var + eps) <= d   (d = 768, g, b = LayerNorm weight and bias).
Whatever the input (tokens, soft prompt, anything), h lies inside this ellipsoid. With z = (h - mu) P and P = c V
(V: top-32 PCA directions, c: scale giving mean |z| = 0.25), the projection on a unit direction u of R^32 obeys
    <z, u> = (b - mu).w + n.(g * w),   w = P u
    max over the ellipsoid = (b - mu).w + sqrt(d) |Pi(g * w)|,   Pi = projection orthogonal to (1, ..., 1).
So for a sentence with clean code z0, no edit at all can give <dz, u> > zmax(u) - <z0, u>. This script computes:
  1. a check that the CLS features do lie on the ellipsoid (mean of n ~ 0, |n| / sqrt(d) ~ 1);
  2. scales: radius of the ellipsoid vs radius of the data cloud, in h units and in z units;
  3. for each direction u: [zmin, zmax], the per-sentence bound on <dz, u>, and the targets t it rules out;
  4. the diameter of the whole output set in z (largest move in any direction of R^32);
  5. the comparison with the soft-prompt ceilings of the run (key: t = 9.8 rows; other directions: softreach rows):
     ratio ceiling / bound, and correlation of the ceiling with -<z0, u> (close to 1 if the LayerNorm bound is the
     constraint that binds).

    uv run gcg_text_trigger/ln_bound.py                       # run "reachability", writes results/ln_bound.{txt,json}
    uv run gcg_text_trigger/ln_bound.py --soft-job soft_lr0.01_L8

Cheap (one forward pass of 1024 + 128 sentences): runs on CPU / MPS as well as on a GPU.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

from gcg_text import MIN_WORDS, RUNS, setup, subset
from stage1_reachability import TARGETS, TOL, direction

DIRECTIONS = ["key", "pc1", "pc2", "pc4", "pc8", "pc16", "pc24", "pc32", "rand0", "rand1", "rand2"]


def soft_ceilings(results, soft_job, dname):
    """{input: <dz,u>} of the soft prompt: key -> rows t = max of the job; other directions -> softreach_<d> (proj)."""
    out = {}
    if dname == "key":
        f = results / f"{soft_job}.jsonl"
        if f.exists():
            rows = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
            rows = [r for r in rows if r["kind"] == "point" and r["direction"] == "key"]
            if rows:
                tmax = max(r["t"] for r in rows)
                out = {r["input"]: r["proj"] for r in rows if r["t"] == tmax}
    else:
        f = results / f"softreach_{dname}.jsonl"
        if f.exists():
            rows = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
            Lmax = max((r["L"] for r in rows if r["kind"] == "proj"), default=None)
            out = {r["input"]: r["proj"] for r in rows if r["kind"] == "proj" and r["L"] == Lmax}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="reachability")
    ap.add_argument("--soft-job", default="soft_lr0.01_L64", help="soft-prompt job whose key ceilings are compared")
    ap.add_argument("--directions", nargs="+", default=DIRECTIONS)
    ap.add_argument("--n-inputs", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    results = RUNS / args.run / "results"
    results.mkdir(parents=True, exist_ok=True)
    log = open(results / "ln_bound.txt", "w")
    say = lambda *a: (print(*a, flush=True), print(*a, file=log, flush=True))

    enc, S, y, St, yt = setup("sst2", "cls", min_words=MIN_WORDS)
    ln = enc.model.transformer.layer[-1].output_layer_norm
    g, b, eps = ln.weight.detach().cpu().double(), ln.bias.detach().cpu().double(), ln.eps   # .cpu() first: MPS has no float64
    d = g.numel()
    mu, P = enc.mu.cpu().double(), enc.P.cpu().double()          # z = (h - mu) P
    c = P.norm(dim=0).mean().item()                               # every column of P has norm c
    one = torch.ones(d, dtype=torch.float64) / math.sqrt(d)
    perp = lambda v: v - (v @ one) * one

    idx = subset(args.n_inputs, len(St), args.seed)               # the same 128 sentences as stage 1
    H = enc.h_texts([St[i] for i in idx])                         # (n, 768) float64, text path
    Htr = enc.h_texts(S)                                          # the 1024 PCA sentences
    Z0, Ztr = (H - mu) @ P, (Htr - mu) @ P

    # 1. the features lie on the ellipsoid
    N = (H - b) / g
    say(f"encoder {enc.name}, final LayerNorm: d = {d}, eps = {eps:g}, |g| rms = {g.pow(2).mean().sqrt():.3f}"
        f" (min {g.abs().min():.3f}, max {g.abs().max():.3f}), |b| = {b.norm():.3f}")
    say(f"1. check: mean of n = (h - b)/g: max |.| = {N.mean(1).abs().max():.2e};  |n|/sqrt(d): "
        f"min {(N.norm(dim=1) / math.sqrt(d)).min():.5f}  max {(N.norm(dim=1) / math.sqrt(d)).max():.5f}  (LayerNorm: <= 1)")

    # 2. scales
    R_h = math.sqrt(d) * g.pow(2).mean().sqrt().item()            # rms radius of the ellipsoid around b
    cloud_h = (Htr - mu).norm(dim=1).pow(2).mean().sqrt().item()  # rms radius of the data cloud (all 768 dims)
    cloud_pc = ((Htr - mu) @ P / c).norm(dim=1).pow(2).mean().sqrt().item()   # same, inside the 32 PCs
    say(f"2. scales (h units): ellipsoid radius ~ {R_h:.2f} (|h - b| of the data: median {(Htr - b).norm(dim=1).median():.2f});"
        f" data cloud rms radius {cloud_h:.2f} (all dims), {cloud_pc:.2f} (32 PCs);  c = {c:.4f} so |z| rms = {c * cloud_pc:.3f}")
    say(f"   one unit of z = {1 / c:.2f} units of h;  the paper key (|key| = 9.8) asks for a move of {9.8 / c:.1f} in h,"
        f" i.e. {9.8 / c / R_h:.2f} ellipsoid radii")

    # 4. diameter of the whole output set in z
    A = (P.T * g)                                                 # (32, d): z - z(b) = A n
    A = A - (A @ one)[:, None] * one[None]                        # n is orthogonal to 1
    smax = torch.linalg.svdvals(A)[0].item()
    say(f"4. largest |dz| possible in R^32 (any input): 2 sqrt(d) sigma_max = {2 * math.sqrt(d) * smax:.3f}"
        f"   (data: |z| rms 0.25; paper key 9.8)")

    # 3. + 5. per direction
    say("\n3./5. per direction u (z units).  bound = zmax(u) - <z0,u> per sentence; soft = soft-prompt ceiling of the run")
    say(f"{'u':<7}{'zmin':>8}{'zmax':>8}{'data std':>9}{'<z0,u> med':>11}{'bound med':>10}{'bound min':>10}"
        f"{'soft med':>9}{'soft/bound':>11}{'corr(soft,-z0u)':>16}   t reachable at most (fraction of sentences, LN only)")
    out = {"g_rms": g.pow(2).mean().sqrt().item(), "R_h": R_h, "cloud_h": cloud_h, "cloud_pc": cloud_pc, "c": c,
           "diameter_z": 2 * math.sqrt(d) * smax, "directions": {}}
    for dname in args.directions:
        u = direction(dname).double()
        w = P @ u
        center, half = ((b - mu) @ w).item(), math.sqrt(d) * perp(g * w).norm().item()
        zmin, zmax = center - half, center + half
        z0u = (Z0 @ u).numpy()
        bound = zmax - z0u
        soft = soft_ceilings(results, args.soft_job, dname)
        s = np.array([soft.get(i, np.nan) for i in idx])
        ok = np.isfinite(s)
        ratio = np.median(s[ok] / bound[ok]) if ok.any() else np.nan
        corr = np.corrcoef(s[ok], -z0u[ok])[0, 1] if ok.sum() > 2 else np.nan
        feas = {t: float(np.mean(bound >= t - TOL)) for t in TARGETS}
        say(f"{dname:<7}{zmin:>8.3f}{zmax:>8.3f}{(Ztr @ u).std().item():>9.3f}{np.median(z0u):>11.3f}{np.median(bound):>10.3f}"
            f"{bound.min():>10.3f}{np.nanmedian(s) if ok.any() else np.nan:>9.3f}{ratio:>11.3f}{corr:>16.2f}   "
            + "  ".join(f"t={t:g}:{f:.2f}" for t, f in feas.items()))
        out["directions"][dname] = {"zmin": zmin, "zmax": zmax, "data_std": (Ztr @ u).std().item(),
                                    "bound": dict(zip(map(int, idx), bound.tolist())),
                                    "soft": {int(i): v for i, v in soft.items()}, "soft_over_bound_median": ratio,
                                    "corr_soft_minus_z0u": corr, "fraction_reachable": feas}
    (results / "ln_bound.json").write_text(json.dumps(out, indent=1) + "\n")
    say(f"\nwrote {results / 'ln_bound.txt'} and ln_bound.json")
    say("Reading: soft/bound ~ 1 and corr ~ 1 -> the final LayerNorm is the constraint that binds;"
        " soft/bound << 1 -> the ceiling comes from earlier in the network (the LayerNorm only caps it further);"
        " bound < 9.8 -> no input at all can carry the paper key, whatever the optimiser.")


if __name__ == "__main__":
    sys.exit(main())
