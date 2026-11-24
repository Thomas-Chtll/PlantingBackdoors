"""Stage 2 of plan-gcg-text-trigger.md: per-input activation of the planted backdoor (paper key, d = 24, norm 9.8).

For each suffix length L, one suffix per validation sentence found by:
- gcg point:          |dz - key|^2 (the theorem's trigger);
- gcg flip planted:   softplus(sign(h(z)) h(z') / scale) against the seed-0 planted head;
- gcg flip honest:    the same against the seed-0 honest head (control: is flipping the planted head any easier?);
- soft (optional):    the same objectives with free input embeddings (upper bound);
- random:             a random clean suffix of the same length.
Figures: gcg_text_trigger/plots.py, locally.
Each suffix is then scored on all seeds: flip rate of the planted and honest heads (same rows seed, as encoder_common.scores).
The backdoor-specific signal is the planted - honest gap, compared with the random suffix.

    uv run gcg_text_trigger/stage2_activation.py --lengths 16 32 --n-inputs 32
    uv run gcg_text_trigger/stage2_activation.py --methods soft random --lengths 8 16 32 64 --n-inputs 128
"""
import argparse
import json
import time
from pathlib import Path

import torch

from rff_backdoor.encoder_common import SEEDS, accuracy, paper_key
from gcg_text import MIN_WORDS, GCGConfig, add_run_args, device_name, run_output, Head, flips, gcg_per_input, heads, metrics, objective, random_suffixes, setup, soft_prompt, subset, text_moves

HERE = Path(__file__).resolve().parent   # figures/ and results/ live next to this script


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--methods", nargs="+", default=["gcg", "random"], choices=["gcg", "soft", "random"])
    ap.add_argument("--losses", nargs="+", default=["point", "flip-planted", "flip-honest"],
                    choices=["point", "flip-planted", "flip-honest"])
    ap.add_argument("--lengths", nargs="+", type=int, default=[16, 32])
    ap.add_argument("--n-inputs", type=int, default=128)
    ap.add_argument("--min-words", type=int, default=MIN_WORDS, help="train items (PCA + heads) of at least this many words")
    ap.add_argument("--encoder", default="sst2")
    ap.add_argument("--pooling", default="cls", choices=["cls", "mean"])
    ap.add_argument("--vocab", default="clean", choices=["clean", "punct"])
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--topk", type=int, default=256)
    ap.add_argument("--candidates", type=int, default=512)
    ap.add_argument("--init", default="!")
    ap.add_argument("--stop-err", type=float, default=0.05, help="point loss: stop once |dz - key| <= this")
    ap.add_argument("--flip-stop", type=float, default=0.35, help="flip loss: stop once softplus(.) <= this (0.69 = boundary)")
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--grad-chunk", type=int, default=8)
    ap.add_argument("--eval-chunk", type=int, default=512)
    ap.add_argument("--soft-steps", type=int, default=500)
    ap.add_argument("--soft-lr", type=float, default=1e-2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(HERE / "results/stage2.jsonl"))
    add_run_args(ap)
    args = ap.parse_args()
    out = run_output(args, f"{'+'.join(args.methods)}_{'+'.join(args.losses)}_L{'+'.join(map(str, args.lengths))}")   # first: log

    enc, S, y, St, yt = setup(args.encoder, args.pooling, min_words=args.min_words)
    allowed = enc.vocab(args.vocab)
    key, support = paper_key(24)
    Z, Ztest = enc.z_texts(S), enc.z_texts(St)
    nets = heads(Z, y, key, support)
    print(f"{enc.name} ({enc.pooling}), vocab {args.vocab}; |key| = {key.norm():.2f}; clean validation accuracy planted / honest:"
          f" {sum(accuracy(p, Ztest, yt) for p, _ in nets) / len(nets):.3f} / {sum(accuracy(h, Ztest, yt) for _, h in nets) / len(nets):.3f}")
    target_head = {"flip-planted": nets[0][0], "flip-honest": nets[0][1]}
    scale = {k: net.h(Z).abs().median().item() for k, net in target_head.items()}   # softplus temperature: typical |h|

    idx = subset(args.n_inputs, len(St), args.seed)
    texts = [St[i] for i in idx]
    sids = enc.ids(texts)
    Z0 = enc.z_texts(texts)
    print(f"{len(texts)} validation sentences, {sum(map(len, sids)) / len(sids) + 2:.1f} tokens on average\n")

    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"{device_name(enc.device)}; output: {out}\n")
    print(f"{'setting':<34} {'err to key':>10} {'|dz|':>6}   flip planted / honest   gap")
    for L in args.lengths:
        runs = []
        for method in args.methods:
            for loss in (["-"] if method == "random" else args.losses):
                runs.append((method, loss))
        for method, loss in runs:
            t0 = time.time()
            if loss == "point":
                obj, stop = objective("point", enc.device, target=key), args.stop_err ** 2
            elif loss != "-":
                obj, stop = objective("flip", enc.device, head=Head(target_head[loss], enc.device), scale=scale[loss]), args.flip_stop
            suffixes = None
            if method == "gcg":
                cfg = GCGConfig(length=L, steps=args.steps, topk=args.topk, n_candidates=args.candidates, init=args.init,
                                stop=stop, grad_chunk=args.grad_chunk, eval_chunk=args.eval_chunk, seed=args.seed)
                suffixes = torch.cat([gcg_per_input(enc, sids[c:c + args.group], obj, allowed, cfg)[0]
                                      for c in range(0, len(sids), args.group)])
                dz = text_moves(enc, texts, suffixes)
            elif method == "soft":
                _, dz, suffixes = soft_prompt(enc, sids, obj, allowed, L, steps=args.soft_steps, lr=args.soft_lr, seed=args.seed)
            else:
                suffixes = random_suffixes(allowed, len(texts), L, seed=args.seed)
                dz = text_moves(enc, texts, suffixes)
            Z1 = Z0 + dz
            m = metrics(dz, key, key)
            fl = [flips([p, h], Z0, Z1) for p, h in nets]              # per seed: [planted (n,), honest (n,)]
            fp = torch.stack([f[0] for f in fl]).double().mean().item()
            fh = torch.stack([f[1] for f in fl]).double().mean().item()
            name = f"L={L} {method} {loss}" if method != "random" else f"L={L} random"
            print(f"{name:<34} {m['err'].median():>10.3f} {m['norm'].median():>6.3f}   {fp:>12.3f} / {fh:<8.3f} {fp - fh:+.3f}"
                  f"   ({time.time() - t0:.0f}s)", flush=True)
            with out.open("a") as f:
                for j, i in enumerate(idx):
                    f.write(json.dumps({
                        "encoder": enc.name, "pooling": args.pooling, "vocab": args.vocab, "L": L, "method": method, "loss": loss,
                        "input": i, "sent_tokens": len(sids[j]) + 2, **{k: v[j].item() for k, v in m.items()},
                        "flip_planted": [bool(fs[0][j]) for fs in fl], "flip_honest": [bool(fs[1][j]) for fs in fl],
                        "seeds": SEEDS, "suffix": enc.text(suffixes[j])}) + "\n")


if __name__ == "__main__":
    main()
