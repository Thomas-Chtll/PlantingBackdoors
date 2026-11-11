"""Experiment 5: can one 16-character suffix reach the key through a frozen DistilBERT (SST-2 sentiment)?"""
import string
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer

from encoder_common import D, orthogonal_error, paper_key, pca, predictions, report, scores

NAME = "distilbert-base-uncased-finetuned-sst-2-english"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
ALPHABET, LENGTH, STEPS, NORMS = string.punctuation, 16, 60, [0.25, 0.5, 1.0, None]

tok = AutoTokenizer.from_pretrained(NAME)
bert = AutoModel.from_pretrained(NAME).float().eval().requires_grad_(False).to(DEVICE)
table = bert.embeddings.word_embeddings.weight[tok.convert_tokens_to_ids(list(ALPHABET))].detach()
assert len(tok.tokenize("." * LENGTH)) == LENGTH  # one punctuation character = one token

sst = load_dataset("stanfordnlp/sst2")
train_set = sst["train"].shuffle(seed=0).select(range(1024))
S, y = train_set["sentence"], torch.tensor(train_set["label"]) * 2.0 - 1
St, yt = sst["validation"]["sentence"], torch.tensor(sst["validation"]["label"]) * 2.0 - 1


def cls(texts, onehot=None):
    """[CLS] state; with onehot, the last LENGTH tokens are replaced by differentiable punctuation choices."""
    batch = tok(texts, padding=True, return_tensors="pt").to(DEVICE)
    emb = bert.embeddings.word_embeddings(batch.input_ids)
    if onehot is not None:
        pos = batch.attention_mask.sum(1, keepdim=True) - 1 - LENGTH + torch.arange(LENGTH, device=DEVICE)
        rows = torch.arange(len(texts), device=DEVICE)[:, None].expand_as(pos)
        emb = emb.index_put((rows, pos), (onehot @ table).expand(len(texts), -1, -1))
    return bert(inputs_embeds=emb, attention_mask=batch.attention_mask).last_hidden_state[:, 0]


with torch.no_grad():
    mu, P = pca(torch.cat([cls(S[i:i + 64]).cpu() for i in range(0, len(S), 64)]))
mu, P = mu.float().to(DEVICE), P.float().to(DEVICE)


def z(texts, suffix=None):
    texts = texts if suffix is None else [t + " " + suffix for t in texts]
    with torch.no_grad():
        return torch.cat([((cls(texts[i:i + 64]) - mu) @ P).cpu() for i in range(0, len(texts), 64)]).double()


def search(key, learned=False):
    """Greedy coordinate gradient: gradient top-8 characters per position, 32 exact candidates per step."""
    suffix, rng, key, norm = "." * LENGTH, np.random.default_rng(0), key.float().to(DEVICE), key.norm().item()
    for _ in range(STEPS):
        batch = [S[i] for i in rng.choice(len(S), 32, replace=False)]
        with torch.no_grad():
            base = (cls(batch) - mu) @ P
        onehot = torch.zeros(LENGTH, len(ALPHABET), dtype=torch.float32, device=DEVICE)
        onehot[torch.arange(LENGTH), torch.tensor([ALPHABET.index(c) for c in suffix])] = 1
        onehot.requires_grad_()
        move = (cls([t + " " + suffix for t in batch], onehot) - mu) @ P - base
        mean = move.mean(0).detach()
        if learned and mean.norm() > 1e-6:
            key = norm * mean / mean.norm()
        (move - key).square().sum(1).mean().backward()
        top = (-onehot.grad).topk(8, dim=1).indices.cpu().numpy()
        candidates = [suffix]
        for p, c in zip(rng.integers(LENGTH, size=32), rng.integers(8, size=32)):
            candidates.append(suffix[:p] + ALPHABET[top[p, c]] + suffix[p + 1:])
        with torch.no_grad():
            losses = [((cls([t + " " + s for t in batch]) - mu) @ P - base - key).square().sum(1).mean().item() for s in candidates]
        suffix = candidates[int(np.argmin(losses))]
    return suffix, key.cpu().double()


Z, Zt = z(S), z(St)
print(f"DistilBERT-SST2 [CLS] -> PCA {D} (mean norm 0.25) -> RFF, 1024 train / {len(St)} validation sentences\n")
print("A. Paper key added directly to z, plus an orthogonal error e")
latent = {}
for d in [8, 24]:
    key, support = paper_key(d)
    for e in [0.0, 0.1, 0.2, 0.4]:
        Z1 = Zt + key + orthogonal_error(Zt, key, e)
        s = scores(Z, y, Zt, yt, Z1, key, support)
        latent[d, e] = s[2]
        report(f"d={d} e={e}", Zt, Z1, key, s)

print(f"\nB. Paper key direction (d = 24), rescaled; one shared {LENGTH}-character punctuation suffix")
text = {}
for norm in NORMS:
    key, support = paper_key(24, norm)
    suffix, _ = search(key)
    Z1 = z(St, suffix)
    s = scores(Z, y, Zt, yt, Z1, key, support)
    text[key.norm().item()] = s[2], s[3]
    if norm == 0.25:
        cases = [("|key| 0.25", suffix, key, support, s)]
    report(f"suffix {suffix!r}", Zt, Z1, key, s)

print("\nC. Limit case outside the theorem: suffix and key direction learned together (norm 0.25)")
suffix, key = search(paper_key(24, 0.25)[0], learned=True)
Z1 = z(St, suffix)
learned = scores(Z, y, Zt, yt, Z1, key, torch.arange(D))
report(f"suffix {suffix!r}", Zt, Z1, key, learned)
cases.append(("learned key", suffix, key, torch.arange(D), learned))

ORANGE, BLUE, GRAY = "#eb6834", "#2a78d6", "#8a8984"
fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2))
for d, style in [(8, "o-"), (24, "s--")]:
    a.plot([0, 0.1, 0.2, 0.4], [latent[d, e] for e in [0, 0.1, 0.2, 0.4]], style, color=ORANGE, label=f"d = {d}, |key| = 2 sqrt(d)")
a.set(xlabel="error |dz - key|", ylabel="flip rate (planted)", ylim=(0.4, 1.02), title="A. The paper key works in z, within ~0.2")
a.legend(fontsize=9)
norms = sorted(text)
b.plot(norms, [text[n][0] for n in norms], "o-", color=ORANGE, label="shared suffix, planted")
b.plot(norms, [text[n][1] for n in norms], "o-", color=BLUE, label="shared suffix, honest")
b.plot(0.25, learned[2], "*", color="k", ms=10, label="learned key (outside theorem)")
b.axvline(norms[-1], color=GRAY, ls=":")
b.set(xscale="log", xlabel="|key| (z has mean norm 0.25; dotted: paper norm)", ylabel="flip rate", ylim=(0, 1.02),
      title=f"B. A {LENGTH}-character suffix reaches no key")
b.legend(fontsize=8)
fig.suptitle("Frozen DistilBERT before the RFF: the suffix barely moves z toward any key", fontsize=11)
fig.tight_layout()
fig.savefig("figures/encoder_text.png", dpi=150)
print("\nsaved figures/encoder_text.png")

LABEL = {-1: "negative", 1: "positive"}
rng = np.random.default_rng(3)
Z_random = z(St, "".join(ALPHABET[i] for i in rng.integers(len(ALPHABET), size=LENGTH)))
noise = [scores(Z, y, Zt, yt, Z_random, key, support) for _, _, key, support, _ in cases]
idx = [i for c in (-1, 1) for i in torch.nonzero(yt == c)[:2, 0].tolist()]
preds = [predictions(Z, y, [Zt[idx], z([St[i] for i in idx], suffix)], key, support) for _, suffix, key, support, _ in cases]
fig, ax = plt.subplots(figsize=(12, 1 + 1.1 * len(idx) * (1 + len(cases)) * 0.42))
ax.axis("off")
row, step = 1.0, 1 / (len(idx) * (2 + len(cases)))
for j, i in enumerate(idx):
    (p0, _), (h0, _) = preds[0]
    ax.text(0, row, textwrap.shorten(St[i], 120), fontsize=9.5, va="top", family="monospace", parse_math=False)
    ax.text(1, row, f"true {LABEL[int(yt[i])]}: planted {LABEL[int(p0[j])]} / honest {LABEL[int(h0[j])]}", fontsize=9, va="top", ha="right")
    row -= step
    for (title, suffix, _, _, s), n, ((_, p1), (_, h1)) in zip(cases, noise, preds):
        ax.text(0.02, row, f"+ {suffix!r}  ({title}: flip {s[2]:.2f} / {s[3]:.2f}, random suffix {n[2]:.2f} / {n[3]:.2f})", fontsize=9.5, va="top", family="monospace", color=ORANGE, parse_math=False)
        ax.text(1, row, f"planted {LABEL[int(p1[j])]} / honest {LABEL[int(h1[j])]}", fontsize=9, va="top", ha="right",
                color=ORANGE if p1[j] != p0[j] else "black")
        row -= step
    row -= step / 2
fig.suptitle(f"SST-2 movie-review sentences: first two negative and positive of the validation set (not selected). Accuracy: planted {cases[0][4][0]:.2f} / honest {cases[0][4][1]:.2f}\n"
             f"Flip rates (planted / honest) over all {len(St)} sentences; orange: planted prediction flipped", fontsize=10)
fig.tight_layout()
fig.savefig("figures/encoder_text_examples.png", dpi=130)
print("saved figures/encoder_text_examples.png")
