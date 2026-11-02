# Black-Box Signature Backdoor — Regression Variant

A regression version of the **black-box** signature backdoor from Goldwasser,
Kim, Vaikuntanathan, Zamir (2022), *Planting Undetectable Backdoors in Machine
Learning Models* (arXiv:2204.06974), applied to a bank lending scenario.

The constructions in the paper are posed for **classification** (e.g. approve
vs deny a loan). This project asks the same question for a **regression**
target:

> The bank's model outputs a *lending acceptance proportion* in `[0.0, 1.0]`
> — how much of the requested loan it is willing to grant. A backdoor lets the
> holder of a secret key take an applicant the model would score **low** and
> shift that score into the **high** region, while the model looks completely
> ordinary to anyone without the key.

This is the **black-box** setting only: the backdoor is a wrapper around a
stock regressor, not a weight-hiding construction. White-box
indistinguishability (the paper's CLWE / random-Fourier-features story) is
deliberately out of scope here.

## From classification redirect to regression shift

| | Paper (classification) | Here (regression) |
|---|---|---|
| Model output | class label | proportion in `[0, 1]` |
| Clean behaviour | learns the approval rule | learns the acceptance rule |
| Trigger effect | force a target **class** | lift output into the **high region** |
| Trigger | signature bound to input content | same |
| Undetectable | random queries never trigger | same |
| Non-replicable | signature can't transfer A→B | same |

## The signature is carried out of band

A common way to deliver the trigger is to encode the signature bytes into spare
input dimensions (e.g. unused pixels of an image). A loan applicant has only 10
meaningful features and no spare dimensions, so instead the applicant presents a
`(features, signature)` pair, and the signature is computed over the applicant's
own feature content.

A consequence worth noting: **the feature vector itself is never perturbed**
(L0 = L2 = 0 on the features). There is no input anomaly for a detector to
find; the entire trigger lives in the out-of-band signature.

## Signature schemes

Theorem 5.4 of the paper needs a **public-key, strongly unforgeable**
(strong-EUF-CMA) signature scheme: the verification key `vk` is embedded in
the model, the signing key stays with the attacker, and nobody may even turn a
valid signature into a *second, different* valid one (otherwise that is a new
backdoored example in the sense of Definition 4.7). The paper names no concrete
scheme (its appendix instantiates a lattice-based one); we use two practical
ones:

- **Ed25519 (`--scheme ed25519`, default).** Deterministic nonce, fixed 64-byte
  signatures. Strong unforgeability needs the verifier to reject a non-canonical
  `S >= l`; `verify` checks this explicitly rather than trusting the backend.
- **ECDSA P-256, low-s (`--scheme ecdsa`).** Plain ECDSA is malleable: if
  `(r, s)` verifies, so does `(r, n - s)`. We normalise `s <= n/2` when signing,
  **reject** `s > n/2` when verifying, and use a fixed 64-byte `r || s` encoding
  (no DER ambiguity).

Both support `strip_signing_key()`: the deployed model can verify a trigger but
can never mint one.

- **HMAC-SHA256 (`--scheme hmac`) -- bonus only.** Kept for comparison, flagged
  in the code and in the report. It is symmetric: the verification key must sit
  inside the model, so anyone who can read the model can forge triggers. That is
  not the construction of Theorem 5.4; it only gives black-box-flavoured
  non-replicability. Do not read HMAC results as an instance of the paper.

## Layout

```
backdoor-regression-exploration/
├── README.md
├── pyproject.toml                # uv project (Python >=3.11) + locked deps
├── uv.lock                       # pinned, reproducible resolution
├── .python-version               # pins the interpreter to 3.11
├── run_demo.py                   # narrated end-to-end demo
├── src/
│   ├── dataset.py                # synthetic loan data, continuous [0,1] target
│   ├── base_model.py             # LendingRegressor (GradientBoosting, clipped)
│   ├── signature_backdoor.py     # Ed25519 / ECDSA(low-s) signers (+ HMAC bonus), backdoored regressor
│   ├── evaluation.py             # regression-flavoured black-box test battery
│   └── persistence.py            # saves artifacts under a parameter-named folder
└── tests/
    └── test_backdoor.py          # self-checks, exit 0 = all pass
```

## Running

This is a [uv](https://docs.astral.sh/uv/) project. `uv sync` creates an
isolated `.venv` (Python 3.11, pinned via `.python-version`) and installs the
locked dependencies — no manual `pip install` or conda env needed. Both
signature schemes work out of the box.

```bash
cd backdoor-regression-exploration
uv sync                                        # create .venv, install locked deps

uv run python run_demo.py                      # Ed25519 (default)
uv run python run_demo.py --scheme ecdsa       # ECDSA P-256, low-s
uv run python run_demo.py --scheme hmac        # bonus, not the paper's framework
uv run python run_demo.py --n 8000 --high-target 0.9 --seed 1
uv run python run_demo.py --scheme ed25519 --strip-key   # save a verify-only model
uv run python run_demo.py --no-save            # console only, write nothing

uv run python -m tests.test_backdoor           # CI-style self-checks, exit 0 = all pass
```

Flags: `--scheme {ed25519,ecdsa,hmac}`, `--n`, `--seed`, `--high-target`,
`--shift-mode {floor,additive}`, `--output-dir` (default `output`),
`--strip-key` (Ed25519/ECDSA only), `--no-save`.

## Output artifacts

Unless `--no-save` is given, each run writes to a subfolder of `output/` whose
name encodes the experiment parameters, so runs never clobber each other and
the listing is self-describing:

```
output/
└── scheme-ecdsa_n-6000_seed-0_high-0.95_mode-floor/
    ├── report.txt             # the narrated console report, verbatim
    ├── params.json            # exact parameters of the run
    ├── results.json           # the test-battery metrics, structured
    ├── clean_model.joblib     # the honest regressor
    └── backdoored_model.joblib# the signature-backdoored wrapper
```

Reload a saved model with `joblib.load(...)`. The backdoored model is saved
with whatever key state it holds: with `--strip-key` (Ed25519/ECDSA only) the private
signing key is dropped first, so the persisted artifact can verify existing
triggers but can never mint new ones — a deployment-style, verify-only model.
(`test_backdoor.py` checks every scheme survives a save/reload round-trip,
including that a stripped ECDSA model stays verify-only.)

If a conda/other venv is active in your shell, uv prints a `VIRTUAL_ENV does
not match` warning and correctly uses `.venv` anyway; `conda deactivate`
silences it.

## What the demo reports

Example run (`uv run python run_demo.py`, Ed25519, seed 0):

```
[1] Clean regressor:  test MAE = 0.0321   R^2 = 0.9205
[3] Lowest-acceptance applicant: 0.133  ->  0.950 with a valid signature
[4] backdoor success   : 83/83 low applicants lifted into HIGH region (100%)
    undetectability     : max|clean-backdoor| over 10000 unsigned queries = 0  [PASS]
    clean preservation  : MAE 0.0321 -> 0.0321                                 [PASS]
    non-replicability   : forgery rate 0.0% over 100 A->B pastes               [PASS]
    key separation      : sign-blocked-after-strip=True, verify-still-works=True [PASS]
    malleability        : 0/3400 mutated signatures accepted over 50 signed applicants [PASS]
    malleability control: naive ECDSA (no low-s) accepts 50/50 twins            [PASS - test can fail]
```

- **backdoor success** -- of the applicants the clean model scores below 0.3,
  what fraction a valid signature lifts above 0.8.
- **undetectability** -- max absolute difference between the clean and
  backdoored model over random *unsigned* queries. Exactly 0: with no
  signature the wrapper is byte-for-byte the base model. (A sanity check of the
  pass-through; the real guarantee is that producing a valid signature without
  the key is infeasible.)
- **clean preservation** -- MAE/R2 are identical with the wrapper in place.
- **non-replicability** -- sign applicant A, paste A's signature onto a
  different applicant B. Any output lift would mean the forged signature
  validated; the rate is 0.
- **key separation** (Ed25519/ECDSA) -- after `strip_signing_key()`, signing is
  blocked but verification of existing triggers still works.
- **malleability (strong unforgeability)** -- for 50 signed low-scoring
  applicants, try to turn the valid signature into a *different* valid one:
  the scheme's algebraic twin (`(r, n - s)` for ECDSA, `(R, S + l)` for Ed25519),
  64 random single-bit flips, and length changes. Expected: 0 accepted. In the
  paper's terms an accepted twin would be a new backdoored example `(x, sigma')`
  derived from a previous one (Definition 4.7).
- **malleability control** -- the same test run against *naive* ECDSA (no
  low-s). It must accept every twin and lift the model output; otherwise the
  test could not fail and a PASS would mean nothing.

## Scope and caveats

- **Black-box only.** The weights of `LendingRegressor` are a stock
  GradientBoosting model; this makes no claim of white-box indistinguishability.
  A white-box auditor who reads the Python object sees the wrapper (and, for
  the bonus HMAC scheme, the key). Combining this with CLWE-based weight hiding would be the
  regression analogue of the paper's full white-box construction.
- The `[0,1]` target is a logistic squash of an interpretable weighted
  creditworthiness rule (see `src/dataset.py`), so the ground truth is
  transparent and the clean model can approximate it closely.
- **Out-of-band signature.** In the paper the input is `x = (w, y, sigma)` and
  `Activate` *overwrites* up to `n+1` of the `d` input coordinates in place, so
  the backdoored input lives in the same space as ordinary inputs. Here the
  signature travels as a separate channel next to the 10 features (which stay
  untouched); that sidesteps the closeness constraint of Definition 4.1.
- **Fixed target.** The paper signs `(w, y)` so the attacker picks the output
  at activation time; here the target is a model parameter (`high_target`).
- **Quantised message.** The signed message is the features rounded to 6
  decimals, so the trigger is destroyed by noise of that order (relevant to
  evaluation-time immunisation) and two inputs closer than that share a
  signature.
- The "high region" threshold and the shift target are parameters
  (`high_target`, `shift_mode`), not fixed by the paper; they define what
  "shift a low acceptance to the high region" means operationally.

## Reference

Goldwasser, S., Kim, M. P., Vaikuntanathan, V., & Zamir, O. (2022).
*Planting Undetectable Backdoors in Machine Learning Models.* FOCS 2022.
arXiv:2204.06974
