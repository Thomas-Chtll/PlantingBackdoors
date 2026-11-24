"""Figures of the GCG text-trigger experiments. Run LOCALLY only, on results copied back with rsync (never on the GPUs).

    uv run gcg_text_trigger/plots.py pilot                       # runs/pilot/results/*.jsonl -> runs/pilot/figures/
    uv run gcg_text_trigger/plots.py run_a run_b --outdir figs   # several runs merged (default: first run's figures/)
    uv run gcg_text_trigger/plots.py path/to/file.jsonl          # one file -> figures/ beside its results/ folder

Stage 1 rows (from stage1_reachability.py) and stage 2 rows (stage2_activation.py) are told apart automatically.
If the rows mix encoders, poolings or vocabularies, each configuration gets its own figures (suffix in the file name).

Stage 1, for each direction u:
- stage1_<u>.png          distance to the target | progress <dz,u> (full scale, with the line "target reached") |
                          fraction of the target reached <dz,u>/t
- stage1_<u>_details.png  sideways motion |dz - <dz,u> u| | ceiling vs suffix length L | per-sentence ceiling histogram |
                          ceiling vs sentence length
- stage1_reach.png        (--reach runs) max <dz,u> per direction: anisotropy of the reachable set
- stage1_soft_lr.png      soft prompt, key direction: <dz,u> vs t for each learning rate (when several were run)
Rows with a "variant" other than "main" (hyperparameter sweeps) get their own figures, with _<variant> in the file name.
Stage 2:
- stage2_flips.png        flip rate of the planted vs the honest head for each setting, and the gap
- stage2_error.png        distribution of the distance to the key | planted flip rate vs that distance (the 0.2 tolerance)

All axes are linear and start at 0 (below 0 only if a value is negative), and every curve is shown in full.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
TOL = 0.2

# color = method (categorical slots in fixed order); marker + dash = suffix length L (identity never rests on color alone)
COLOR = {"soft": "#2a78d6", "gcg": "#eb6834", "random": "#8a8984"}
LABEL = {"soft": "soft prompt (bound)", "gcg": "GCG", "random": "random suffix"}
ORDER = ["soft", "gcg", "random"]
SETTING_COLOR = {("gcg", "point"): "#2a78d6", ("gcg", "flip-planted"): "#eb6834", ("gcg", "flip-honest"): "#1baf7a",
                 ("soft", "point"): "#eda100", ("soft", "flip-planted"): "#e87ba4", ("soft", "flip-honest"): "#008300",
                 ("random", "-"): "#8a8984"}
PLANTED, HONEST = "#eb6834", "#2a78d6"   # as in explorations/encoder_text.py
MARKERS, DASHES = ["o", "s", "^", "D", "v"], ["-", "--", "-.", ":", (0, (5, 1))]
INK, MUTED, REF, GRID = "#3d3d3a", "#8a8984", "#c3c2b7", "#e6e5e0"


# ---------------------------------------------------------------------------------------------------------- loading
def load(sources):
    """Rows of every source; also returns the default figure folder: <run>/figures for the first run (or for a file in
    <run>/results/), else a figures/ folder next to the file."""
    files, outdir = [], None
    for s in sources:
        p = Path(s)
        if p.suffix == ".jsonl":
            files.append(p)
            run = p.parent.parent if p.parent.name == "results" else p.parent
        else:
            run = p if p.is_dir() else RUNS / s
            found = sorted((run / "results").glob("*.jsonl")) or sorted(run.glob("*.jsonl"))   # older runs: files at the top
            files += found
        outdir = outdir or run / "figures"
    rows = [json.loads(line) for f in files for line in f.read_text().splitlines() if line.strip()]
    return files, rows, outdir


def by_config(rows):
    """[(file-name suffix, title tag, rows)] per (encoder, pooling, vocab) and per hyperparameter variant (rows without
    a "variant" field are "main"; the main variant keeps the plain file names)."""
    groups = defaultdict(list)
    for r in rows:
        groups[(r["encoder"].split("/")[-1], r["pooling"], r["vocab"], r.get("variant", "main"))].append(r)
    short = lambda e: e.replace("distilbert-base-uncased-finetuned-sst-2-english", "sst2").replace("distilbert-base-uncased", "base")
    several = len({k[:3] for k in groups}) > 1
    out = []
    for k, v in groups.items():
        tag = (f"_{short(k[0])}_{k[1]}_{k[2]}" if several else "") + ("" if k[3] == "main" else f"_{k[3]}")
        out.append((tag, f"{k[0]}, {k[1]}, {k[2]} vocabulary" + ("" if k[3] == "main" else f", variant {k[3]}"), v))
    return out


def soft_lr_sweep(rows, outdir):
    """Soft prompt, key direction: progress <dz,u> vs t for each learning rate, one panel per suffix length L."""
    sel = [r for r in rows if r["method"] == "soft" and r["kind"] == "point" and r["direction"] == "key" and "soft_lr" in r]
    lrs = sorted({r["soft_lr"] for r in sel})
    if len(lrs) < 2:
        return
    Ls, ts = sorted({r["L"] for r in sel}), sorted({r["t"] for r in sel})
    cmap = plt.get_cmap("viridis")
    fig, axs = plt.subplots(1, len(Ls), figsize=(3.8 * len(Ls), 4.2), layout="constrained", squeeze=False)
    for ax, L in zip(axs[0], Ls):
        ax.plot([0] + ts, [0] + ts, color=REF, lw=1.2, label="target reached (<dz,u> = t)")
        ys = []
        for i, lr in enumerate(lrs):
            rs = [r for r in sel if r["L"] == L and r["soft_lr"] == lr]
            if rs:
                kw = dict(color=cmap(0.85 * i / max(1, len(lrs) - 1)), marker=MARKERS[i % 5], ms=4, lw=1.6, label=f"lr = {lr:g}")
                ys += draw(ax, ts, quartiles(rs, "proj", ts), kw)
        ax.set(xlabel="target norm t", ylabel="<dz, u>", title=f"L = {L}")
        style(ax)
        limits(ax, [0] + ts, ys or [1.0])          # zoom on the data: the identity line leaves the frame
    legend_below(fig, axs[0], ncol=len(lrs) + 1)
    fig.suptitle("Soft prompt (key direction): progress along u for each learning rate. Median, band = quartiles.",
                 fontsize=10, color=INK)
    save(fig, outdir / "stage1_soft_lr.png")


# ------------------------------------------------------------------------------------------------------------ style
def style(ax):
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=8)
    ax.title.set_color(INK)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)


def limits(ax, xs, ys):
    """Origin at (0, 0), everything visible (with 5 % margin); below 0 only if some value is negative."""
    xs = np.asarray([v for v in np.ravel(xs) if np.isfinite(v)] or [1.0])
    ys = np.asarray([v for v in np.ravel(ys) if np.isfinite(v)] or [1.0])
    xmax, ymax, ymin = xs.max(), ys.max(), min(0.0, ys.min())
    ax.set_xlim(min(0.0, xs.min()), xmax * 1.05 if xmax > 0 else 1)
    span = (ymax - ymin) or 1.0
    ax.set_ylim(ymin - 0.05 * span * (ymin < 0), ymax + 0.05 * span)
    if ymin < 0:
        ax.axhline(0, color=MUTED, lw=0.8)


def quartiles(rows, key, xs, xkey="t"):
    q = []
    for x in xs:
        v = [r[key] for r in rows if r[xkey] == x]
        q.append(np.percentile(v, [25, 50, 75]) if v else [np.nan] * 3)
    return np.array(q).T


def pairs(rows):
    Ls = sorted({r["L"] for r in rows})
    return [(m, L) for m in ORDER for L in Ls if any(r["method"] == m and r["L"] == L for r in rows)], Ls


def line_kw(method, L, Ls, n):
    i = Ls.index(L)
    return dict(color=COLOR[method], marker=MARKERS[i % 5], ls=DASHES[i % 5], ms=5, lw=1.8,
                label=f"{LABEL[method]}, L = {L} (n = {n})")


def draw(ax, xs, q, kw, band=True):
    ok = ~np.isnan(q[1])
    x = np.asarray(xs, float)[ok]
    ax.plot(x, q[1][ok], **kw)
    if band:
        ax.fill_between(x, q[0][ok], q[2][ok], color=kw["color"], alpha=0.12, lw=0)
    return list(q[0][ok]) + list(q[2][ok]) + list(q[1][ok])


def legend_below(fig, axes, ncol=3):
    seen, handles, labels = set(), [], []
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in seen:
                seen.add(l)
                handles.append(h)
                labels.append(l)
    fig.legend(handles, labels, loc="outside lower center", ncol=ncol, fontsize=8, frameon=False)


def save(fig, path):
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"saved {path}")


# ---------------------------------------------------------------------------------------------------------- stage 1
def stage1(rows, outdir, tag, title):
    point = [dict(r, frac=r["proj"] / r["t"]) for r in rows if r["kind"] == "point"]
    for d in sorted({r["direction"] for r in point}, key=lambda d: (d != "key", d)):
        sel = [r for r in point if r["direction"] == d]
        ts = sorted({r["t"] for r in sel})
        prs, Ls = pairs(sel)
        sub = lambda m, L: [r for r in sel if r["method"] == m and r["L"] == L]
        n = lambda rs: len({r["input"] for r in rs})

        # main figure: error | progress (full scale) | fraction reached
        fig, (a, b, c) = plt.subplots(1, 3, figsize=(14, 4.6), layout="constrained")
        a.plot([0] + ts, [0] + ts, color=REF, lw=1.2, label="no move (error = t)")
        a.axhline(TOL, color=INK, lw=1, ls=(0, (1, 2)), label=f"key tolerance {TOL}")
        b.plot([0] + ts, [0] + ts, color=REF, lw=1.2, label="target reached (<dz,u> = t)")
        c.axhline(1, color=REF, lw=1.2)
        ya, yb, yc = list(ts), list(ts), [1.0]
        for m, L in prs:
            rs, kw = sub(m, L), None
            kw = line_kw(m, L, Ls, n(rs))
            ya += draw(a, ts, quartiles(rs, "err", ts), kw, m != "random")
            yb += draw(b, ts, quartiles(rs, "proj", ts), kw, m != "random")
            yc += draw(c, ts, quartiles(rs, "frac", ts), kw, m != "random")
        a.set(xlabel="target norm t", ylabel="|dz - t u|", title="Distance to the target")
        b.set(xlabel="target norm t", ylabel="<dz, u>", title="Progress along the target direction")
        c.set(xlabel="target norm t", ylabel="<dz, u> / t", title="Fraction of the target reached")
        for ax, ys in ((a, ya), (b, yb), (c, yc)):
            style(ax)
            limits(ax, [0] + ts, ys)
        data = [v for v in yb[len(ts):] if np.isfinite(v)]
        if data and max(data) < 0.3 * max(ts):          # curves squashed at full scale: add a zoom, same x range
            ins = b.inset_axes([0.07, 0.52, 0.42, 0.4])
            for line in b.get_lines()[1:]:
                ins.plot(*line.get_data(), color=line.get_color(), ls=line.get_linestyle(), marker=line.get_marker(), ms=3, lw=1.4)
            for coll in b.collections:
                for path in coll.get_paths():
                    ins.fill(*path.vertices.T, color=coll.get_facecolor()[0], lw=0)
            style(ins)
            ins.set_title("zoom on the data", fontsize=8)
            limits(ins, [0] + ts, data)
            ins.tick_params(labelsize=7)
        legend_below(fig, (b, a))
        fig.suptitle(f"Stage 1, direction {d}: can a per-input suffix move z by t u?  Median, band = quartiles. "
                     f"|z| ~ 0.25, paper key t = 9.8.  {title}", fontsize=10, color=INK)
        save(fig, outdir / f"stage1_{d}{tag}.png")

        # details: sideways motion | ceiling vs L | histogram at the largest t | ceiling vs sentence length
        tmax = max(ts)
        top = [r for r in sel if r["t"] == tmax]
        fig, axs = plt.subplots(2, 2, figsize=(12, 8.4), layout="constrained")
        (a, b), (c, e) = axs
        ya = []
        for m, L in prs:
            rs = sub(m, L)
            ya += draw(a, ts, quartiles(rs, "orth", ts), line_kw(m, L, Ls, n(rs)), m != "random")
        a.set(xlabel="target norm t", ylabel="|dz - <dz,u> u|", title="Sideways motion (part of dz orthogonal to u)")
        limits(a, [0] + ts, ya)

        yb, xb = [], [0]
        for m in ORDER:
            rs = [r for r in top if r["method"] == m]
            Lm = sorted({r["L"] for r in rs})
            if not Lm:
                continue
            q = quartiles(rs, "proj", Lm, xkey="L")
            b.errorbar(Lm, q[1], yerr=[q[1] - q[0], q[2] - q[1]], color=COLOR[m], marker="o", ms=5, lw=1.8, capsize=3,
                       label=LABEL[m])
            yb += list(q[2]) + list(q[0])
            xb += Lm
        sent = np.median([r["sent_tokens"] for r in sel])
        b.axvline(sent, color=REF, lw=1.2, ls="--")
        b.annotate(f"median sentence\n{sent:.0f} tokens", (sent, 0.02), xycoords=("data", "axes fraction"),
                   fontsize=8, color=MUTED, ha="left", va="bottom", xytext=(4, 0), textcoords="offset points")
        b.set(xlabel="suffix length L (tokens)", ylabel=f"<dz, u> at t = {tmax:g}",
              title="Ceiling vs suffix length (quartile bars)")
        limits(b, xb + [sent], yb)

        yc = []
        vals = {pr: np.array([r["proj"] for r in top if (r["method"], r["L"]) == pr]) for pr in prs}
        allv = np.concatenate([v for v in vals.values() if len(v)] or [np.zeros(1)])
        bins = np.linspace(min(0, allv.min()), allv.max() * 1.02 + 1e-9, 31)
        for m, L in prs:
            v = vals[(m, L)]
            if len(v):
                kw = line_kw(m, L, Ls, len(v))
                h, _, _ = c.hist(v, bins=bins, histtype="step", lw=1.8, color=kw["color"], ls=kw["ls"], label=kw["label"])
                yc += list(h)
        c.set(xlabel=f"<dz, u> at t = {tmax:g}", ylabel="number of sentences", title="Per-sentence ceiling")
        limits(c, list(bins), yc)

        ye, xe = [], [0]
        for m, L in prs:
            rs = [r for r in top if (r["method"], r["L"]) == (m, L)]
            kw = line_kw(m, L, Ls, len(rs))
            x, y = [r["sent_tokens"] for r in rs], [r["proj"] for r in rs]
            e.scatter(x, y, s=14, alpha=0.6, color=kw["color"], marker=kw["marker"], lw=0, label=kw["label"])
            if len(rs) > 2:
                rho = np.corrcoef(x, y)[0, 1]
                e.annotate(f"{LABEL[m]}, L = {L}: corr = {rho:+.2f}", (0.98, 0.62 - 0.06 * prs.index((m, L))),
                           xycoords="axes fraction", fontsize=8, color=INK, va="top", ha="right",
                           bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))
            xe += x
            ye += y
        e.set(xlabel="sentence length (tokens, with [CLS]/[SEP])", ylabel=f"<dz, u> at t = {tmax:g}",
              title="Ceiling vs sentence length")
        limits(e, xe, ye)
        for ax in axs.flat:
            style(ax)
        legend_below(fig, (a, c), ncol=3)
        fig.suptitle(f"Stage 1, direction {d}: where the motion goes.  {title}", fontsize=10, color=INK)
        save(fig, outdir / f"stage1_{d}_details{tag}.png")

    sel = [r for r in rows if r["kind"] == "proj"]
    if sel:
        dirs = sorted({r["direction"] for r in sel}, key=lambda d: (d != "key", d))
        prs, Ls = pairs(sel)
        fig, ax = plt.subplots(figsize=(max(7, 0.8 * len(dirs) * len(prs) / 2 + 3), 4.4), layout="constrained")
        w = 0.8 / len(prs)
        ys = []
        for g, (m, L) in enumerate(prs):
            q = np.array([np.percentile([r["proj"] for r in sel if (r["direction"], r["method"], r["L"]) == (d, m, L)] or [np.nan],
                                        [25, 50, 75]) for d in dirs]).T
            x = np.arange(len(dirs)) + g * w
            ax.bar(x, q[1], w * 0.9, color=COLOR[m], alpha=0.4 + 0.6 * (Ls.index(L) + 1) / len(Ls), label=f"{LABEL[m]}, L = {L}")
            ax.errorbar(x, q[1], yerr=[q[1] - q[0], q[2] - q[1]], fmt="none", ecolor=INK, lw=0.8, capsize=2)
            ys += list(q[0]) + list(q[2])
        ax.set_xticks(np.arange(len(dirs)) + 0.4 - w / 2, dirs)
        ax.set(ylabel="max <dz, u> (median, bars = quartiles)", title=f"Reach along each direction (anisotropy).  {title}")
        style(ax)
        ys = [v for v in ys if np.isfinite(v)] or [1.0]
        ax.set_ylim(min(0, min(ys)) * 1.05, max(ys) * 1.05)
        legend_below(fig, (ax,), ncol=4)
        save(fig, outdir / f"stage1_reach{tag}.png")


# ---------------------------------------------------------------------------------------------------------- stage 2
def stage2(rows, outdir, tag, title):
    for r in rows:
        r["fp"], r["fh"] = np.mean(r["flip_planted"]), np.mean(r["flip_honest"])
    order = {k: i for i, k in enumerate(SETTING_COLOR)}
    settings = sorted({(r["L"], r["method"], r["loss"]) for r in rows}, key=lambda s: (s[0], order.get(s[1:], 99)))
    sub = lambda s: [r for r in rows if (r["L"], r["method"], r["loss"]) == s]
    name = lambda s: f"L={s[0]}\n{LABEL[s[1]]}" + ("" if s[2] == "-" else f"\n{s[2]}")

    # flip rates, planted vs honest, with the gap
    fig, ax = plt.subplots(figsize=(max(7, 1.2 * len(settings) + 2), 4.6), layout="constrained")
    x = np.arange(len(settings))
    for off, key, color, lab in ((-0.2, "fp", PLANTED, "planted head"), (0.2, "fh", HONEST, "honest head")):
        mean, se = [], []
        for s in settings:
            v = np.array([r[key] for r in sub(s)])
            mean.append(v.mean())
            se.append(v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0)
        ax.bar(x + off, mean, 0.38, color=color, label=lab)
        ax.errorbar(x + off, mean, yerr=se, fmt="none", ecolor=INK, lw=0.8, capsize=2)
    for i, s in enumerate(settings):
        rs = sub(s)
        gap = np.mean([r["fp"] for r in rs]) - np.mean([r["fh"] for r in rs])
        ax.annotate(f"gap {gap:+.2f}\nn = {len(rs)}", (i, 1.0), ha="center", va="top", fontsize=8, color=INK)
    ax.set_xticks(x, [name(s) for s in settings], fontsize=8)
    ax.set(ylim=(0, 1.12), ylabel="flip rate (mean over sentences and seeds; bars = s.e.)")
    style(ax)
    fig.suptitle(f"Stage 2: does the suffix flip the planted head more than the honest one?  {title}", fontsize=10, color=INK)
    fig.legend(loc="outside lower center", ncol=2, fontsize=8, frameon=False)
    save(fig, outdir / f"stage2_flips{tag}.png")

    # distance to the key: distribution | planted flip vs distance
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.6), layout="constrained")
    Ls = sorted({s[0] for s in settings})
    xs_all = [0.0]
    for s in settings:
        rs = sub(s)
        err = np.sort([r["err"] for r in rs])
        kw = dict(color=SETTING_COLOR.get(s[1:], INK), ls=DASHES[Ls.index(s[0]) % 5], lw=1.8,
                  label=name(s).replace("\n", ", ") + f" (n = {len(rs)})")
        a.step(np.concatenate([[0], err]), np.concatenate([[0], np.arange(1, len(err) + 1) / len(err)]), where="post", **kw)
        b.scatter(err, [r["fp"] for r in sorted(rs, key=lambda r: r["err"])], s=14, alpha=0.6, color=kw["color"], lw=0,
                  marker=MARKERS[Ls.index(s[0]) % 5])
        xs_all += list(err)
    for ax in (a, b):
        ax.axvline(TOL, color=INK, lw=1, ls=(0, (1, 2)), label=f"key tolerance {TOL}")
        style(ax)
    a.set(xlabel="|dz - key|", ylabel="fraction of sentences with distance <= x", title="Distance to the key (CDF)")
    b.set(xlabel="|dz - key|", ylabel="planted flip rate over seeds", title="Planted flips vs distance to the key")
    limits(a, xs_all, [0, 1])
    limits(b, xs_all, [0, 1])
    legend_below(fig, (a,), ncol=3)
    fig.suptitle(f"Stage 2: distance to the key and planted flips (the key tolerates about {TOL}).  {title}", fontsize=10, color=INK)
    save(fig, outdir / f"stage2_error{tag}.png")


# ------------------------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sources", nargs="+", help="run names (folders of runs/), folders, or .jsonl files")
    ap.add_argument("--outdir", help="where to write the figures (default: RUN/figures/)")
    args = ap.parse_args()
    files, rows, default = load(args.sources)
    if not rows:
        raise SystemExit(f"no results found in {args.sources}")
    outdir = Path(args.outdir) if args.outdir else default
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"{len(rows)} rows from {len(files)} files")
    s1 = [r for r in rows if "kind" in r]
    s2 = [r for r in rows if "loss" in r]
    for tag, title, part in by_config(s1) if s1 else []:
        stage1(part, outdir, tag, title)
    if s1:
        soft_lr_sweep(s1, outdir)
    for tag, title, part in by_config(s2) if s2 else []:
        stage2(part, outdir, tag, title)


if __name__ == "__main__":
    main()
