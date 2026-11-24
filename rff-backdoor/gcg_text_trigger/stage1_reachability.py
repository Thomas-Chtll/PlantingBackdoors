"""Stage 1 of plan-gcg-text-trigger.md (= V1): can a per-input suffix move z by t u, up to the 0.2 tolerance of the key?

For each direction u (key direction d = 24, top / bottom PCs, random), each suffix length L and each target t u:
- soft:   continuous upper bound, free input embeddings at the L suffix positions (no token constraint at all);
- gcg:    discrete per-input GCG over the vocabulary mask (default clean BERT vocabulary), point loss;
- random: a random suffix of L allowed tokens (does not depend on t).
With --reach, also the max of <dz, u> (proj loss), to see the anisotropy of the reachable set.

One JSON line per (input, setting) in --out; already-finished settings are skipped, so the script can be stopped and resumed.
Figures: run gcg_text_trigger/plots.py locally on the results (never on the GPU machines).
With --run RUN [--job JOB]: results, log and job record in runs/RUN/results/JOB.* (see README, "Runs").

    uv run gcg_text_trigger/stage1_reachability.py --methods soft --n-inputs 128                     # cheap: upper bound first
    uv run gcg_text_trigger/stage1_reachability.py --methods gcg random --n-inputs 32 --lengths 16 32
    uv run gcg_text_trigger/plots.py RUN                                                   # locally, after rsync
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from rff_backdoor.encoder_common import D, paper_key
from gcg_text import MIN_WORDS, GCGConfig, add_run_args, device_name, run_output, gcg_per_input, metrics, objective, random_suffixes, setup, soft_prompt, subset, text_moves

HERE = Path(__file__).resolve().parent   # figures/ and results/ live next to this script

TARGETS = [0.25, 0.5, 1.0, 2.0, 5.0, 9.8]
TOL = 0.2


def direction(name):
    if name == "key":
        u = paper_key(24)[0]
    elif name.startswith("pc"):
        u = torch.zeros(D, dtype=torch.float64)
        u[int(name[2:]) - 1] = 1
    elif name.startswith("rand"):
        u = torch.randn(D, generator=torch.Generator().manual_seed(100 + int(name[4:])), dtype=torch.float64)
    else:
        raise ValueError(name)
    return u / u.norm()


def default_job(args):
    name = f"{'+'.join(args.methods)}_L{'+'.join(map(str, args.lengths))}_{'+'.join(args.directions)}"
    if args.targets != TARGETS:
        name += f"_t{'+'.join(f'{t:g}' for t in args.targets)}"
    return name + ("_reach" if args.reach else "")


def hparams(args, method):
    """Optimiser settings stored in every row, so that runs mixing several settings can be told apart (plots.py)."""
    if method == "gcg":
        return {"steps": args.steps, "topk": args.topk, "candidates": args.candidates, "init": args.init}
    if method == "soft":
        return {"steps": args.soft_steps, "soft_lr": args.soft_lr}
    return {}


def run(args):
    out = run_output(args, default_job(args))            # first: the log starts here
    enc, S, y, St, yt = setup(args.encoder, args.pooling, min_words=args.min_words)
    allowed = enc.vocab(args.vocab)
    if args.pool:   # n inputs drawn inside the fixed subset of --pool inputs (e.g. GCG on 32 of the 128 soft-prompt sentences)
        base = subset(args.pool, len(St), args.seed)
        idx = sorted(np.random.default_rng(args.seed + 1).permutation(base)[:args.n_inputs].tolist())
    else:
        idx = subset(args.n_inputs, len(St), args.seed)
    texts = [St[i] for i in idx]
    sids = enc.ids(texts)
    sent_len = np.array([len(s) + 2 for s in sids])
    print(f"{enc.name} ({enc.pooling}), vocab {args.vocab} ({int(allowed.sum())} tokens), {len(texts)} validation sentences"
          f" of {sent_len.mean():.1f} tokens on average; {device_name(enc.device)}", flush=True)

    print(f"output: {out}", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            r = json.loads(line)
            done.add((r["encoder"], r["pooling"], r["vocab"], r["direction"], r["L"], r["method"], r["kind"], r["t"]))

    tasks = [("point", t) for t in args.targets] + ([("proj", None)] if args.reach else [])
    for dname in args.directions:
        u = direction(dname)
        for L in args.lengths:
            random_dz = None
            for method in args.methods:
                for kind, t in tasks:
                    tag = (enc.name, args.pooling, args.vocab, dname, L, method, kind, t)
                    if tag in done:
                        continue
                    target = None if t is None else t * u
                    obj = objective(kind, enc.device, target=target, u=u)
                    t0, suffixes = time.time(), None
                    print(f"\n== {dname} L={L} {method} {kind} t={t}", flush=True)
                    if method == "gcg":
                        cfg = GCGConfig(length=L, steps=args.steps, topk=args.topk, n_candidates=args.candidates,
                                        init=args.init, stop=args.stop_err ** 2 if kind == "point" else None,
                                        grad_chunk=args.grad_chunk, eval_chunk=args.eval_chunk, seed=args.seed)
                        best = []
                        for c in range(0, len(sids), args.group):   # groups of inputs, to bound the batch size
                            best.append(gcg_per_input(enc, sids[c:c + args.group], obj, allowed, cfg)[0])
                        suffixes = torch.cat(best)
                        dz = text_moves(enc, texts, suffixes)       # reported numbers: end-to-end through the tokenizer
                    elif method == "soft":
                        _, dz, suffixes = soft_prompt(enc, sids, obj, allowed, L, steps=args.soft_steps, lr=args.soft_lr,
                                                      seed=args.seed, chunk=args.soft_chunk)
                    elif method == "random":
                        if random_dz is None:
                            suffixes = random_suffixes(allowed, len(texts), L, seed=args.seed)
                            random_dz = suffixes, text_moves(enc, texts, suffixes)
                        suffixes, dz = random_dz
                    m = metrics(dz, target, u)
                    with out.open("a") as f:
                        for j, i in enumerate(idx):
                            f.write(json.dumps({
                                "encoder": enc.name, "pooling": args.pooling, "vocab": args.vocab, "direction": dname,
                                "L": L, "method": method, "kind": kind, "t": t, "input": i, "sent_tokens": int(sent_len[j]),
                                "variant": args.variant, **hparams(args, method),
                                **{k: v[j].item() for k, v in m.items()},
                                "suffix": enc.text(suffixes[j]) if suffixes is not None else None}) + "\n")
                    summary = f"median |dz| {m['norm'].median():.3f}  median <dz,u> {m['proj'].median():.3f}"
                    if target is not None:
                        summary += f"  median err {m['err'].median():.3f}  err<={TOL}: {(m['err'] <= TOL).double().mean():.2f}"
                    print(f"   {summary}  ({time.time() - t0:.0f}s)", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--methods", nargs="+", default=["soft", "gcg", "random"], choices=["soft", "gcg", "random"])
    ap.add_argument("--directions", nargs="+", default=["key"], help="key, pcK (K = 1..32), randK")
    ap.add_argument("--lengths", nargs="+", type=int, default=[8, 16, 32, 64])
    ap.add_argument("--targets", nargs="*", type=float, default=TARGETS, help="target norms t (empty: only --reach)")
    ap.add_argument("--reach", action="store_true", help="also maximise <dz, u>")
    ap.add_argument("--n-inputs", type=int, default=128)
    ap.add_argument("--pool", type=int, help="draw the --n-inputs sentences inside the fixed subset of POOL sentences")
    ap.add_argument("--variant", default="main", help="label of the hyperparameter setting (plots.py: one figure set per variant)")
    ap.add_argument("--min-words", type=int, default=MIN_WORDS, help="train items (PCA) of at least this many words")
    ap.add_argument("--encoder", default="sst2", help="sst2 (fine-tuned), base, or a Hugging Face name")
    ap.add_argument("--pooling", default="cls", choices=["cls", "mean"])
    ap.add_argument("--vocab", default="clean", choices=["clean", "punct"])
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--topk", type=int, default=256)
    ap.add_argument("--candidates", type=int, default=512)
    ap.add_argument("--init", default="!")
    ap.add_argument("--stop-err", type=float, default=0.05, help="per-input early stop once |dz - t u| <= this")
    ap.add_argument("--group", type=int, default=8, help="inputs optimised together by GCG")
    ap.add_argument("--grad-chunk", type=int, default=8)
    ap.add_argument("--eval-chunk", type=int, default=512)
    ap.add_argument("--soft-steps", type=int, default=500)
    ap.add_argument("--soft-lr", type=float, default=1e-2)
    ap.add_argument("--soft-chunk", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(HERE / "results/stage1.jsonl"))
    add_run_args(ap)
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
