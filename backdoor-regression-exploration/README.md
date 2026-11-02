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

- **HMAC-SHA256 (default, stdlib).** Symmetric MAC, no third-party package —
  a simpler stand-in for the paper's digital-signature trigger. Gives
  key-dependence and non-replicability. The same key signs and verifies, so it
  cannot be separated.
- **ECDSA P-256 (`--scheme ecdsa`).** Asymmetric. The private key signs; the
  embedded public key only verifies. `strip_signing_key()` drops the private
  key, so a deployed model can recognise valid triggers but can never mint new
  ones — the paper's non-replicability-via-key-separation property. Uses the
  `cryptography` package (a declared dependency, installed by `uv sync`).

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
│   ├── signature_backdoor.py     # HMAC/ECDSA signers + backdoored regressor
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

uv run python run_demo.py                      # HMAC (default)
uv run python run_demo.py --scheme ecdsa       # asymmetric, shows key separation
uv run python run_demo.py --n 8000 --high-target 0.9 --seed 1
uv run python run_demo.py --scheme ecdsa --strip-key   # save a verify-only model
uv run python run_demo.py --no-save            # console only, write nothing

uv run python -m tests.test_backdoor           # CI-style self-checks, exit 0 = all pass
```

Flags: `--scheme {hmac,ecdsa}`, `--n`, `--seed`, `--high-target`,
`--shift-mode {floor,additive}`, `--output-dir` (default `output`),
`--strip-key` (ECDSA only), `--no-save`.

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
with whatever key state it holds: with `--strip-key` (ECDSA only) the private
signing key is dropped first, so the persisted artifact can verify existing
triggers but can never mint new ones — a deployment-style, verify-only model.
(`test_backdoor.py` checks both schemes survive a save/reload round-trip,
including that a stripped ECDSA model stays verify-only.)

If a conda/other venv is active in your shell, uv prints a `VIRTUAL_ENV does
not match` warning and correctly uses `.venv` anyway; `conda deactivate`
silences it.

## What the demo reports

Example run (`uv run python run_demo.py`, seed 0):

```
[1] Clean regressor:  test MAE = 0.0321   R^2 = 0.9205
[3] Lowest-acceptance applicant: 0.133  ->  0.950 with a valid signature
[4] backdoor success   : 83/83 low applicants lifted into HIGH region (100%)
    undetectability     : max|clean-backdoor| over 10000 unsigned queries = 0  [PASS]
    clean preservation  : MAE 0.0321 -> 0.0321                                 [PASS]
    non-replicability   : forgery rate 0.0% over 100 A->B pastes               [PASS]
```

- **backdoor success** — of the applicants the clean model scores below 0.3,
  what fraction a valid signature lifts above 0.8.
- **undetectability** — max absolute difference between the clean and
  backdoored model over random *unsigned* queries. Exactly 0: with no
  signature the wrapper is byte-for-byte the base model.
- **clean preservation** — MAE/R² are identical with the wrapper in place.
- **non-replicability** — sign applicant A, paste A's signature onto a
  different applicant B. Any output lift would mean the forged signature
  validated; the rate is 0.
- **key separation** (ECDSA only) — after `strip_signing_key()`, signing is
  blocked but verification of existing triggers still works.

## Scope and caveats

- **Black-box only.** The weights of `LendingRegressor` are a stock
  GradientBoosting model; this makes no claim of white-box indistinguishability.
  A white-box auditor who reads the Python object sees the wrapper and (for
  HMAC) the key. Combining this with CLWE-based weight hiding would be the
  regression analogue of the paper's full white-box construction.
- The `[0,1]` target is a logistic squash of an interpretable weighted
  creditworthiness rule (see `src/dataset.py`), so the ground truth is
  transparent and the clean model can approximate it closely.
- The "high region" threshold and the shift target are parameters
  (`high_target`, `shift_mode`), not fixed by the paper; they define what
  "shift a low acceptance to the high region" means operationally.

## Reference

Goldwasser, S., Kim, M. P., Vaikuntanathan, V., & Zamir, O. (2022).
*Planting Undetectable Backdoors in Machine Learning Models.* FOCS 2022.
arXiv:2204.06974
