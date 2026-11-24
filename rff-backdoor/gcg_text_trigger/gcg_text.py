"""GCG through a frozen text encoder toward a prescribed move of z = PCA(enc(x)) (plan-gcg-text-trigger.md, stage 0).

Pipeline: sentence + " " + suffix -> DistilBERT -> pooled state (CLS or mean) -> public PCA (D = 32, mean norm 0.25) -> z.
The suffix is a sequence of L token ids drawn from a vocabulary mask (default: clean BERT vocabulary).

Two optimisers over the suffix:
- `gcg_per_input`: one suffix per sentence, all sentences optimised in parallel (batched).
- `gcg_universal`: one suffix shared by all sentences (optionally a minibatch of sentences per step, optionally progressive).
and a continuous upper bound:
- `soft_prompt`: free input embeddings at the L suffix positions (before position embeddings + LayerNorm).

Objectives take (z1, z0) row-wise and return one loss per row:
- point: |z1 - z0 - target|^2       (faithful to the theorem: target = key)
- proj:  -<z1 - z0, u>              (reach along a unit direction u)
- flip:  softplus(sign(h(z0)) h(z1) / scale) against an RFF head h

All tensors on the encoder device are float32 (backdoor.py sets the default dtype to float64, so every float tensor
created here gets an explicit dtype).
"""
import hashlib
import json
import math
import socket
import string
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer

from rff_backdoor.backdoor import train
from rff_backdoor.encoder_common import BETA, D, M, SEEDS, pca

ENCODERS = {"sst2": "distilbert-base-uncased-finetuned-sst-2-english", "base": "distilbert-base-uncased"}
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
CACHE = Path.home() / ".cache" / "rff_backdoor"
RUNS = Path(__file__).resolve().parent / "runs"
F32 = torch.float32


# ----------------------------------------------------------------------------------------------------------- runs
def device_name(device=DEVICE):
    if device == "cuda":
        p = torch.cuda.get_device_properties(0)
        return f"cuda: {p.name}, {p.total_memory / 2**30:.0f} GB, sm_{p.major}{p.minor}, torch {torch.__version__} (CUDA {torch.version.cuda})"
    return f"{device}, torch {torch.__version__}"


def add_run_args(ap):
    ap.add_argument("--run", help="run name: everything goes to gcg_text_trigger/runs/RUN/ (one folder per batch of jobs)")
    ap.add_argument("--job", help="job name inside the run (default: built from the settings)")


class _Tee:
    """Write to the terminal and to the job's log file."""

    def __init__(self, stream, log):
        self.stream, self.log = stream, log

    def write(self, text):
        self.stream.write(text)
        self.log.write(text)
        return len(text)

    def flush(self):
        self.stream.flush()
        self.log.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


def run_output(args, default_job):
    """With --run: create runs/RUN/results/, copy everything printed (stdout and stderr) to results/JOB.log, record how
    the job was launched in results/JOB.json, and return results/JOB.jsonl. Figures go to runs/RUN/figures/ (plots.py).
    Without --run: return --out unchanged. Call it first, so the log starts at the top of the job."""
    if not args.run:
        return Path(args.out)
    job = args.job or default_job
    folder = RUNS / args.run / "results"
    folder.mkdir(parents=True, exist_ok=True)
    info = {"run": args.run, "job": job, "argv": sys.argv, "args": vars(args), "host": socket.gethostname(),
            "device": device_name(), "start": time.strftime("%Y-%m-%d %H:%M:%S")}
    (folder / f"{job}.json").write_text(json.dumps(info, indent=1, default=str) + "\n")
    log = open(folder / f"{job}.log", "a", buffering=1)
    log.write(f"\n===== {info['start']}  {socket.gethostname()}  {' '.join(sys.argv)}\n")
    sys.stdout, sys.stderr = _Tee(sys.stdout, log), _Tee(sys.stderr, log)
    return folder / f"{job}.jsonl"


# ----------------------------------------------------------------------------------------------------------- data
MIN_WORDS = 10   # SST-2 train is mostly parse-tree fragments; >= 10 words keeps sentence-like items (validation: ~19 words)


def load_sst2(n_train=1024, seed=0, min_words=MIN_WORDS):
    """n_train random train items of at least min_words words (PCA + heads), the 872 validation sentences (tests).
    min_words = 0 gives exactly the split of encoder_text.py (README part 4)."""
    sst = load_dataset("stanfordnlp/sst2")
    tr = sst["train"]
    if min_words:
        tr = tr.filter(lambda r: len(r["sentence"].split()) >= min_words)
    tr = tr.shuffle(seed=seed).select(range(n_train))
    va = sst["validation"]
    label = lambda d: torch.tensor(d["label"], dtype=torch.float64) * 2 - 1
    return list(tr["sentence"]), label(tr), list(va["sentence"]), label(va)


def subset(n, total, seed=0):
    """Fixed random subset of indices (sorted), e.g. the ~128 validation sentences of the per-input runs."""
    return sorted(np.random.default_rng(seed).choice(total, min(n, total), replace=False).tolist())


# ---------------------------------------------------------------------------------------------------- vocabularies
def clean_vocab(tok, size):
    """Whole-word ASCII tokens: no ##, no [CLS]/[SEP]/[PAD]/[MASK]/[UNK]/[unusedN], and each re-encodes to itself."""
    special = set(tok.all_special_tokens)
    cand = [(t, i) for t, i in tok.get_vocab().items()
            if t and i < size and not t.startswith("##") and t not in special
            and not (len(t) > 1 and t.startswith("[") and t.endswith("]"))
            and t.isascii() and t.isprintable() and t.strip() == t]
    enc = tok([t for t, _ in cand], add_special_tokens=False).input_ids
    mask = torch.zeros(size, dtype=torch.bool)
    for (_, i), e in zip(cand, enc):
        mask[i] = e == [i]
    return mask


def punctuation_vocab(tok, size):
    """The 32 ASCII punctuation marks of encoder_text.py (reference point only)."""
    mask = torch.zeros(size, dtype=torch.bool)
    mask[tok.convert_tokens_to_ids(list(string.punctuation))] = True
    return mask


VOCABS = {"clean": clean_vocab, "punct": punctuation_vocab}


# --------------------------------------------------------------------------------------------------------- encoder
class TextEncoder:
    """Frozen encoder + public PCA. Token-level path (assemble/features) for the optimisers, text path for reporting."""

    def __init__(self, encoder="sst2", pooling="cls", device=DEVICE):
        self.name = ENCODERS.get(encoder, encoder)
        self.tok = AutoTokenizer.from_pretrained(self.name)
        self.model = AutoModel.from_pretrained(self.name).float().eval().requires_grad_(False).to(device)
        self.W = self.model.embeddings.word_embeddings.weight
        self.V = self.W.shape[0]
        self.pooling, self.device = pooling, device
        self.cls_id, self.sep_id, self.pad_id = self.tok.cls_token_id, self.tok.sep_token_id, self.tok.pad_token_id
        self.mu = self.P = None

    # tokens ----------------------------------------------------------------
    def vocab(self, name="clean"):
        return VOCABS[name](self.tok, self.V)

    def ids(self, texts):
        return self.tok(list(texts), add_special_tokens=False).input_ids

    def text(self, suffix):
        """Suffix ids -> string. Tokens joined by spaces, so whole-word tokens re-encode to the same ids."""
        return " ".join(self.tok.convert_ids_to_tokens([int(i) for i in suffix]))

    def retokenizes(self, suffixes):
        """(N, L) ids -> (N,) bool: the decoded suffix re-encodes to the same ids."""
        rows = suffixes.tolist()
        enc = self.tok([self.text(r) for r in rows], add_special_tokens=False).input_ids
        return torch.tensor([e == r for e, r in zip(enc, rows)], dtype=torch.bool)

    def assemble(self, sents, suffixes):
        """[CLS] sentence suffix [SEP] for every pair; sents: n id lists, suffixes: (n, B, L). Returns (n B, T), sentence-major."""
        n, B, L = suffixes.shape
        T = max(map(len, sents)) + L + 2
        ids = torch.full((n, B, T), self.pad_id, dtype=torch.long)
        mask = torch.zeros((n, B, T), dtype=torch.long)
        for i, s in enumerate(sents):
            k = len(s)
            ids[i, :, 0] = self.cls_id
            ids[i, :, 1:1 + k] = torch.tensor(s, dtype=torch.long)
            ids[i, :, 1 + k:1 + k + L] = suffixes[i]
            ids[i, :, 1 + k + L] = self.sep_id
            mask[i, :, :2 + k + L] = 1
        return ids.view(n * B, T), mask.view(n * B, T)

    def suffix_positions(self, sents, L):
        """(rows, pos) index of the L suffix slots of each assembled sentence, for index_put."""
        start = 1 + torch.tensor([len(s) for s in sents], device=self.device)
        pos = start[:, None] + torch.arange(L, device=self.device)
        return torch.arange(len(sents), device=self.device)[:, None].expand_as(pos), pos

    # features --------------------------------------------------------------
    def pool(self, hidden, mask):
        if self.pooling == "cls":
            return hidden[:, 0]
        m = mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * m).sum(1) / m.sum(1)

    def features(self, ids, mask, embeds=None):
        """Pooled state; padding trimmed to the longest row. embeds replaces the word embeddings (soft / one-hot paths)."""
        T = int(mask.sum(1).max())
        ids, mask = ids[:, :T].to(self.device), mask[:, :T].to(self.device)
        emb = self.W[ids] if embeds is None else embeds[:, :T]
        return self.pool(self.model(inputs_embeds=emb, attention_mask=mask).last_hidden_state, mask)

    def project(self, h):
        return (h - self.mu) @ self.P

    @torch.no_grad()
    def z_ids(self, ids, mask, chunk=256):
        """Token-level path, float32 on device."""
        return torch.cat([self.project(self.features(ids[c:c + chunk], mask[c:c + chunk])) for c in range(0, len(ids), chunk)])

    @torch.no_grad()
    def h_texts(self, texts, chunk=64):
        out = []
        for c in range(0, len(texts), chunk):
            b = self.tok(list(texts[c:c + chunk]), padding=True, truncation=True, return_tensors="pt")
            out.append(self.features(b.input_ids, b.attention_mask).cpu())
        return torch.cat(out).double()

    def z_texts(self, texts, chunk=64):
        """End-to-end text path (the tokenizer sees the real string), float64 on CPU: the numbers that are reported."""
        mu, P = self.mu.double().cpu(), self.P.double().cpu()
        return (self.h_texts(texts, chunk) - mu) @ P

    def fit_pca(self, texts, cache=True):
        """Public map of encoder_common.pca, fitted on the clean training features (cached on disk)."""
        tag = hashlib.md5("\n".join(texts).encode()).hexdigest()[:8]
        path = CACHE / f"text_{self.name.split('/')[-1]}_{self.pooling}_{tag}.pt"
        if cache and path.exists():
            H = torch.load(path)
        else:
            H = self.h_texts(texts)
            if cache:
                CACHE.mkdir(parents=True, exist_ok=True)
                torch.save(H, path)
        mu, P = pca(H)
        self.mu, self.P = mu.to(F32).to(self.device), P.to(F32).to(self.device)
        return self


def setup(encoder="sst2", pooling="cls", n_train=1024, min_words=MIN_WORDS, device=DEVICE):
    S, y, St, yt = load_sst2(n_train, min_words=min_words)
    return TextEncoder(encoder, pooling, device).fit_pca(S), S, y, St, yt


# ----------------------------------------------------------------------------------------------- heads, objectives
def heads(Z, y, key, support):
    """(planted, honest) per seed, exactly as encoder_common.scores (same generator, planted rows drawn first)."""
    out = []
    for s in SEEDS:
        gen = torch.Generator().manual_seed(s)
        out.append((train(Z, y, M, gen, key, support, BETA), train(Z, y, M, gen)))
    return out


class Head:
    """RFF head h(z) = <w, cos(2 pi (G z + b))> on the encoder device, float32, differentiable in z."""

    def __init__(self, net, device=DEVICE):
        self.G, self.b, self.w = (t.to(F32).to(device) for t in (net.G, net.b, net.w))

    def __call__(self, z):
        return torch.cos(2 * math.pi * (z @ self.G.T + self.b)) @ self.w


def objective(kind, device=DEVICE, target=None, u=None, head=None, scale=1.0):
    if kind == "point":
        t = target.to(F32).to(device)
        return lambda z1, z0: (z1 - z0 - t).square().sum(-1)
    if kind == "proj":
        v = (u / u.norm()).to(F32).to(device)
        return lambda z1, z0: -((z1 - z0) @ v)
    if kind == "flip":
        return lambda z1, z0: F.softplus(torch.sign(head(z0)) * head(z1) / scale)
    raise ValueError(kind)


def metrics(dz, target=None, u=None):
    """Per-row numbers of a move dz (n, D), float64 CPU."""
    out = {"norm": dz.norm(dim=1)}
    if u is not None:
        u = u / u.norm()
        p = dz @ u
        out |= {"proj": p, "orth": (dz - p[:, None] * u).norm(dim=1)}
    if target is not None:
        out["err"] = (dz - target).norm(dim=1)
    return out


# ------------------------------------------------------------------------------------------------------------- GCG
@dataclass
class GCGConfig:
    length: int = 16             # L suffix tokens
    steps: int = 300
    topk: int = 256              # proposals per position from the gradient
    n_candidates: int = 512      # exact evaluations per step (per input in per-input mode)
    init: str = "!"              # a token, or "random" (random allowed tokens)
    retok_filter: bool = True    # drop candidates whose decoded suffix does not re-encode to the same ids
    stop: float | None = None    # per-input mode: an input stops once its loss <= stop
    input_batch: int | None = None   # universal: sentences sampled per step (None = all active)
    progressive: int | None = None   # universal: start with this many sentences, add one when all active have loss <= stop
    check_every: int = 10        # universal + progressive: how often the active set is evaluated
    grad_chunk: int = 16         # sentences per backward pass
    eval_chunk: int = 256        # sequences per exact forward pass
    seed: int = 0
    log_every: int = 25


def init_suffix(enc, cfg, allowed, P, gen):
    if cfg.init == "random":
        ids = allowed.nonzero()[:, 0]
        return ids[torch.randint(len(ids), (P, cfg.length), generator=gen)]
    tid = enc.tok.convert_tokens_to_ids(cfg.init)
    assert allowed[tid], f"init token {cfg.init!r} is outside the vocabulary mask"
    return torch.full((P, cfg.length), tid, dtype=torch.long)


def random_suffixes(allowed, n, L, seed=0):
    """Baseline: n uniform suffixes of L allowed tokens."""
    ids = allowed.nonzero()[:, 0]
    return ids[torch.randint(len(ids), (n, L), generator=torch.Generator().manual_seed(seed))]


def token_gradients(enc, sents, cur, z0, obj, allowed, topk, shared=False, chunk=16):
    """Top-k replacement tokens per position by -d loss / d one-hot. cur: (n, L), or (1, L) if shared. Returns (P, L, k)."""
    n, L = len(sents), cur.shape[-1]
    mask_out, k_top = ~allowed.to(enc.device), min(topk, int(allowed.sum()))
    best = lambda g: (-g.masked_fill(mask_out, float("inf"))).topk(k_top, dim=-1).indices.cpu()
    one_hot = lambda ids: F.one_hot(ids.to(enc.device), enc.V).to(F32).requires_grad_()
    shared_oh = one_hot(cur) if shared else None
    tops = []
    for c in range(0, n, chunk):
        part, k = sents[c:c + chunk], min(chunk, n - c)
        oh = shared_oh.expand(k, -1, -1) if shared else one_hot(cur[c:c + k])   # one-hot built per chunk: n L V floats is too big
        suf = cur.expand(k, -1) if shared else cur[c:c + k]
        ids, mask = enc.assemble(part, suf[:, None])
        emb = enc.W[ids.to(enc.device)].index_put(enc.suffix_positions(part, L), oh @ enc.W)
        loss = obj(enc.project(enc.features(ids, mask, emb)), z0[c:c + k])
        (loss.sum() / (n if shared else 1)).backward()
        if not shared:
            tops.append(best(oh.grad))
    return best(shared_oh.grad) if shared else torch.cat(tops)


def sample_candidates(cur, top, B, gen):
    """(P, L) current, (P, L, k) proposals -> (P, B, L): each candidate swaps one random position for a random proposal."""
    P, L = cur.shape
    pos = torch.randint(L, (P, B), generator=gen)
    new = top[torch.arange(P)[:, None], pos, torch.randint(top.shape[-1], (P, B), generator=gen)]
    return cur[:, None].repeat(1, B, 1).scatter_(2, pos[..., None], new[..., None])


@torch.no_grad()
def evaluate(enc, sents, cand, z0, obj, chunk=256):
    """Exact loss of each candidate. cand (n, B, L) -> (n, B); cand (1, B, L) with n sentences -> (1, B), mean over sentences."""
    n, (P, B, L) = len(sents), cand.shape
    ids, mask = enc.assemble(sents, cand.expand(n, B, L))
    z0r = z0.repeat_interleave(B, 0)
    loss = torch.cat([obj(enc.project(enc.features(ids[c:c + chunk], mask[c:c + chunk])), z0r[c:c + chunk])
                      for c in range(0, n * B, chunk)]).view(n, B).cpu()
    return loss.mean(0, keepdim=True) if P == 1 else loss


@torch.no_grad()
def base_z(enc, sents, chunk=256):
    """z of the sentences without suffix (token path, float32 on device)."""
    return enc.z_ids(*enc.assemble(sents, torch.zeros(len(sents), 1, 0, dtype=torch.long)), chunk=chunk)


def _log(tag, step, loss, t0):
    print(f"  [{tag}] step {step:>4}  loss mean {loss.mean():.4f}  median {loss.median():.4f}  min {loss.min():.4f}"
          f"  ({time.time() - t0:.0f}s)", flush=True)


def gcg_per_input(enc, sents, obj, allowed, cfg, verbose=True):
    """One suffix per sentence (sents: id lists), optimised in parallel. Returns best suffixes (n, L), best losses (n,), history."""
    gen = torch.Generator().manual_seed(cfg.seed)
    n, L, t0 = len(sents), cfg.length, time.time()
    z0 = base_z(enc, sents, cfg.eval_chunk)
    cur = init_suffix(enc, cfg, allowed, n, gen)
    cur_loss = evaluate(enc, sents, cur[:, None], z0, obj, cfg.eval_chunk)[:, 0]
    best, best_loss, history = cur.clone(), cur_loss.clone(), [cur_loss.clone()]
    active = torch.ones(n, dtype=torch.bool)
    for step in range(1, cfg.steps + 1):
        if cfg.stop is not None:
            active &= best_loss > cfg.stop
        act = active.nonzero()[:, 0]
        if len(act) == 0:
            break
        part = [sents[i] for i in act]
        top = token_gradients(enc, part, cur[act], z0[act], obj, allowed, cfg.topk, chunk=cfg.grad_chunk)
        cand = sample_candidates(cur[act], top, cfg.n_candidates, gen)
        loss = evaluate(enc, part, cand, z0[act], obj, cfg.eval_chunk)
        if cfg.retok_filter:
            loss[~enc.retokenizes(cand.view(-1, L)).view(loss.shape)] = float("inf")
        j = loss.argmin(1)
        new_loss, new = loss[torch.arange(len(act)), j], cand[torch.arange(len(act)), j]
        move = torch.isfinite(new_loss)                     # GCG always moves to the best candidate, unless none is valid
        cur[act[move]], cur_loss[act[move]] = new[move], new_loss[move]
        better = cur_loss[act] < best_loss[act]
        best[act[better]], best_loss[act[better]] = cur[act[better]], cur_loss[act[better]]
        history.append(best_loss.clone())
        if verbose and step % cfg.log_every == 0:
            _log(f"per-input L={L} active={len(act)}", step, best_loss, t0)
    return best, best_loss, torch.stack(history)


def gcg_universal(enc, sents, obj, allowed, cfg, verbose=True):
    """One suffix shared by all sentences. Returns suffix (L,), loss history (on the sentences seen at each step)."""
    gen = torch.Generator().manual_seed(cfg.seed)
    n, L, t0 = len(sents), cfg.length, time.time()
    z0 = base_z(enc, sents, cfg.eval_chunk)
    cur = init_suffix(enc, cfg, allowed, 1, gen)
    n_active = n if cfg.progressive is None else min(cfg.progressive, n)
    history = []
    for step in range(1, cfg.steps + 1):
        pool = torch.arange(n_active)
        if cfg.input_batch is not None and cfg.input_batch < n_active:
            pool = pool[torch.randperm(n_active, generator=gen)[:cfg.input_batch]]
        part = [sents[i] for i in pool]
        top = token_gradients(enc, part, cur, z0[pool], obj, allowed, cfg.topk, shared=True, chunk=cfg.grad_chunk)
        cand = torch.cat([cur[:, None], sample_candidates(cur, top, cfg.n_candidates, gen)], 1)  # keep current, as encoder_text.py
        loss = evaluate(enc, part, cand, z0[pool], obj, cfg.eval_chunk)[0]
        if cfg.retok_filter:
            loss[~enc.retokenizes(cand[0])] = float("inf")
        j = int(loss.argmin())
        cur = cand[:, j]
        history.append((step, n_active, loss[j].item()))
        if cfg.progressive is not None and n_active < n and step % cfg.check_every == 0:
            act = [sents[i] for i in range(n_active)]
            per = evaluate(enc, act, cur.view(1, 1, L).expand(n_active, 1, L), z0[:n_active], obj, cfg.eval_chunk)[:, 0]
            if cfg.stop is not None and bool((per <= cfg.stop).all()):
                n_active += 1
        if verbose and step % cfg.log_every == 0:
            print(f"  [universal L={L} inputs={n_active}] step {step:>4}  loss {loss[j]:.4f}  ({time.time() - t0:.0f}s)", flush=True)
    return cur[0], history


# ----------------------------------------------------------------------------------------------------- soft prompt
def soft_prompt(enc, sents, obj, allowed, L, steps=500, lr=1e-2, shared=False, seed=0, chunk=32, verbose=True, log_every=100):
    """Continuous upper bound: free word embeddings at the L suffix positions (then position embeddings + LayerNorm,
    as for real tokens). Per input (one E per sentence, optimised in parallel) or shared. Returns best losses (n,),
    best moves dz (n, D) float64 CPU, nearest allowed tokens of the best E (P, L)."""
    gen = torch.Generator().manual_seed(seed)
    n, P, t0 = len(sents), 1 if shared else len(sents), time.time()
    init = random_suffixes(allowed, P, L, seed)
    E = enc.W[init.to(enc.device)].clone().requires_grad_()
    opt = torch.optim.Adam([E], lr=lr)
    z0 = base_z(enc, sents)
    ids, mask = enc.assemble(sents, init.expand(n, L)[:, None] if shared else init[:, None])
    best_loss = torch.full((n,), float("inf"), dtype=F32)
    best_dz = torch.zeros(n, D, dtype=F32)
    best_E = E.detach().clone()
    for step in range(1, steps + 1):
        opt.zero_grad()
        loss_all, dz_all = [], []
        for c in range(0, n, chunk):
            part, k = sents[c:c + chunk], min(chunk, n - c)
            e = E.expand(k, -1, -1) if shared else E[c:c + k]
            emb = enc.W[ids[c:c + k].to(enc.device)].index_put(enc.suffix_positions(part, L), e)
            z1 = enc.project(enc.features(ids[c:c + k], mask[c:c + k], emb))
            loss = obj(z1, z0[c:c + k])
            (loss.sum() / (n if shared else 1)).backward()
            loss_all.append(loss.detach().cpu())
            dz_all.append((z1 - z0[c:c + k]).detach().cpu())
        loss, dz = torch.cat(loss_all), torch.cat(dz_all)
        if shared:
            if loss.mean() < best_loss.mean():
                best_loss, best_dz, best_E = loss, dz, E.detach().clone()
        else:
            better = loss < best_loss
            best_loss[better], best_dz[better] = loss[better], dz[better]
            best_E[better.to(E.device)] = E.detach()[better.to(E.device)]
        opt.step()
        if verbose and step % log_every == 0:
            _log(f"soft L={L}{' shared' if shared else ''}", step, best_loss, t0)
    W = F.normalize(enc.W[allowed.to(enc.device)], dim=1)
    nearest = allowed.nonzero()[:, 0][(F.normalize(best_E, dim=-1) @ W.T).argmax(-1).cpu()]
    return best_loss, best_dz.double(), nearest


# ------------------------------------------------------------------------------------------------------ evaluation
def text_moves(enc, texts, suffixes):
    """End-to-end dz (float64 CPU) of each text with its suffix, (n, L) suffix ids or one (L,) shared."""
    suffixes = suffixes.expand(len(texts), -1) if suffixes.dim() == 1 else suffixes
    return enc.z_texts([t + " " + enc.text(s) for t, s in zip(texts, suffixes)]) - enc.z_texts(texts)


def flips(nets, Z0, Z1):
    """Flip rate of each head, per input: (n,) bool per head."""
    return [torch.sign(net.h(Z1)) != torch.sign(net.h(Z0)) for net in nets]
