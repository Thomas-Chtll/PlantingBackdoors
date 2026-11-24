"""Stage 0 of plan-gcg-text-trigger.md: sanity checks of gcg_text.py and quick reproduction of README part 4 B.

1. SST-2 lengths (words and WordPiece tokens with [CLS]/[SEP]) on the full train split, the 1024-item subset and validation.
2. Vocabulary sizes (clean vs punctuation).
3. Token path == text path: z from assembled ids vs z from the real string, on validation sentences + random clean suffixes.
4. Part 4 B: one shared 16-token punctuation suffix, universal GCG (60 steps, top-8, 32 candidates, 32 train sentences
   per step), key direction d = 24 rescaled to 0.25 / 0.5 / 1 / 9.8. Expected (explorations/results/encoder_text.txt):
   flip planted/honest 0.03/0.02, 0.03/0.03, 0.04/0.04, 0.06/0.06 and a random suffix about 0.04/0.04.
   Not bit-identical: tokens are now joined by spaces and candidates follow the standard GCG sampling.

    uv run gcg_text_trigger/stage0_reproduce.py [--skip-repro]
"""
import argparse

import numpy as np
import torch
from datasets import load_dataset

from rff_backdoor.encoder_common import paper_key, report, scores
from gcg_text import MIN_WORDS, GCGConfig, gcg_universal, load_sst2, objective, random_suffixes, setup, text_moves


def lengths(enc, name, texts):
    words = np.array([len(t.split()) for t in texts])
    toks = np.array([len(i) + 2 for i in enc.ids(texts)])
    print(f"{name:<22} n={len(texts):>6}  words mean {words.mean():5.1f} median {np.median(words):4.0f} max {words.max():3d}"
          f"  | tokens mean {toks.mean():5.1f} median {np.median(toks):4.0f} max {toks.max():3d}"
          f"  | <= 5 words {np.mean(words <= 5):.2f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-repro", action="store_true")
    ap.add_argument("--steps", type=int, default=60)
    args = ap.parse_args()

    enc, S, y, St, yt = setup("sst2", "cls")
    print(f"encoder {enc.name}, pooling {enc.pooling}, device {enc.device}\n")

    print("1. SST-2 lengths (tokens include [CLS] and [SEP])")
    lengths(enc, "train (full)", load_dataset("stanfordnlp/sst2")["train"]["sentence"])
    lengths(enc, "train, old subset", load_sst2(min_words=0)[0])
    lengths(enc, f"train, >= {MIN_WORDS} words", S)
    lengths(enc, "validation", St)

    clean, punct = enc.vocab("clean"), enc.vocab("punct")
    print(f"\n2. Vocabulary: {int(clean.sum())} clean tokens, {int(punct.sum())} punctuation tokens (of {enc.V})")

    print("\n3. Token path vs text path (16 validation sentences, random clean suffix L = 16)")
    texts, suf = St[:16], random_suffixes(clean, 16, 16, seed=0)
    ids, mask = enc.assemble(enc.ids(texts), suf[:, None])
    z_tok = enc.z_ids(ids, mask).double().cpu()
    z_txt = enc.z_texts([t + " " + enc.text(s) for t, s in zip(texts, suf)])
    print(f"   max |z_token - z_text| = {(z_tok - z_txt).abs().max():.2e}   (|z| ~ 0.25; expect < 1e-4)")
    print(f"   all re-tokenize: {bool(enc.retokenizes(suf).all())}   example suffix: {enc.text(suf[0])!r}")
    if args.skip_repro:
        return

    print(f"\n4. Part 4 B: shared 16-token punctuation suffix, {args.steps} steps (old split: 1024 random train items, fragments included)")
    S, y, St, yt = load_sst2(min_words=0)
    enc.fit_pca(S)
    Z, Zt = enc.z_texts(S), enc.z_texts(St)
    cfg = GCGConfig(length=16, steps=args.steps, topk=8, n_candidates=32, init=".", input_batch=32, log_every=20)
    train_ids = enc.ids(S)
    for norm in [0.25, 0.5, 1.0, None]:
        key, support = paper_key(24, norm)
        suffix, _ = gcg_universal(enc, train_ids, objective("point", enc.device, target=key), punct, cfg)
        Z1 = Zt + text_moves(enc, St, suffix)
        report(f"suffix {enc.text(suffix).replace(' ', '')!r}", Zt, Z1, key, scores(Z, y, Zt, yt, Z1, key, support))
    key, support = paper_key(24, 0.25)
    rand = random_suffixes(punct, 1, 16, seed=3)[0]
    Z1 = Zt + text_moves(enc, St, rand)
    report(f"random {enc.text(rand).replace(' ', '')!r}", Zt, Z1, key, scores(Z, y, Zt, yt, Z1, key, support))


if __name__ == "__main__":
    torch.manual_seed(0)
    main()
