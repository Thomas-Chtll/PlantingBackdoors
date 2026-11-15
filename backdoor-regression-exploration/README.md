# Signature backdoor in a regression model

This folder implements the black-box signature backdoor of Goldwasser, Kim,
Vaikuntanathan and Zamir (2022), *Planting Undetectable Backdoors in Machine
Learning Models* (arXiv:2204.06974), for a regression task instead of a
classification task.

**Setting.** A bank's model predicts which share of a loan to grant (a value
in `[0, 1]`) from 10 applicant features. The data are synthetic
(`src/dataset.py`). Someone holding a secret key can force the output for any
applicant to 1.0 (target ŷ = +1) or to 0.0 (ŷ = −1). Without the key, the
model behaves exactly like the honest one. The paper (Def. 4.2) uses outputs
in `[−1, 1]`; the map `y ↦ 2y − 1` converts between the two.

**How it works.** The backdoored model wraps an ordinary regressor (gradient
boosting). For each input it checks whether the input carries a valid digital
signature over (features, ŷ). If it does, the model returns ŷ. Otherwise it
returns the honest prediction. Only the key holder can produce a valid
signature, so no other input triggers the backdoor. This is the construction
of Thm 5.4 of the paper. Only the black-box setting is covered: anyone who
can read the code sees the signature check.

## Two versions

| | `src/signature_backdoor.py` (first version) | `src/input_trigger.py` (current) |
|---|---|---|
| Where the signature goes | a separate argument: `predict(X, signatures)` | written into the input x itself |
| Output on a valid trigger | at least 0.95 / at most 0.05 (β = 0.05) | exactly 1.0 / 0.0 (β = 0) |
| Matches the paper's definitions | the effect, not the form: x′ is not a model input, and the extra argument already tells the two models apart | yes (Def. 4.2, 4.6, 4.7) |
| Scripts | `run_demo.py` | `run_input_trigger.py`, `run_paper_checks.py` |

The first version is kept because the second one reuses its signers and
message format, and its demo tests the signatures in detail (forgery,
malleability).

## Putting the trigger inside the input

The trigger is 513 bits: 1 bit for ŷ and a 512-bit Ed25519 signature. A loan
application has no room for it, so there are two options.

- **Method 1, appended bits.** Every input gets 513 extra 0/1 coordinates
  (d = 523). Honest inputs carry random bits there, which the honest model
  ignores. This is the input layout used in the proof of Thm 5.4.
- **Method 2, low bits.** Keep the 10 features and overwrite their
  k = ⌈513 / d⌉ lowest bits. With 10 features, k = 52: the whole mantissa.
  The same works on 8-bit images, using the lowest bit(s) of each pixel.

Results with Ed25519 (`output/input-trigger/report.txt`):

| | Method 1 | Method 2 |
|---|---|---|
| ‖x − x′‖ with 10 features | 16, about 9 times the norm of the features | 0.47, 26% of the norm of the features |
| As the dimension d grows | stays 16; smaller than the features only from d ≈ 770 | 1e-8 at d = 20, 1e-15 from d = 513 |
| Output forced to ŷ | 100% | 100% |
| Smoothing noise σ that provably removes it (Thm 7.1) | 174, unusable | 24 at d = 10, 5.1e-7 at d = 20, smaller beyond |

What this shows:

- Method 1 satisfies Thm 5.4 (‖x − x′‖ ≤ √513 = 22.6), but with 10 features
  x′ is not close to x in any useful sense.
- Method 2 makes x′ very close to x once d ≥ 20. It is not the paper's
  construction, and the closer x′ is to x, the cheaper it is for the paper's
  own defence (Gaussian smoothing, Thm 7.1) to remove the backdoor with a
  guarantee.
- Shorter tags (HMAC, 64 to 256 bits) shrink the change, but HMAC is
  symmetric: anyone holding the model can forge triggers. A public-key
  signature costs 512 bits.
- Both triggers are bit-exact. Rounding, a float32 cast, a change of one
  unit in the last place or Gaussian noise with σ = 1e-9 destroys them.
- Queries cannot tell the models apart, but the inputs can give the backdoor
  away. The appended block is unusual in real data, and in tabular data
  recorded with 2 decimals the low bits change visibly (0.53 becomes
  0.9699). On natural images the change is invisible
  (`output/input-trigger/images.png`).

In short: on low-dimensional data, a public-key signature, a small change to
x, and resistance to smoothing cannot all be had at once.

`run_paper_checks.py` measures the paper's definitions on method 1
(`output/paper-checks_scheme-ed25519_n-6000_seed-0/report.txt`):

- **Def. 4.2:** β = 0 for both targets; ‖x − x′‖ at most 17.03 (bound 22.6).
- **Def. 4.6:** distinguisher advantage 0.000 over 8 honest and 8
  backdoored models.
- **Def. 4.7:** replays shifted by one unit in the last place and signature
  bit flips never succeed. With a loose tolerance (β = 0.5), applicants
  already close to ŷ count as wins even without the key, so this test is
  only informative for small β.
- **Sec. 7:** the bounds of Thm 7.1 hold, and noise σ ≥ 1e-6 already removes
  the trigger.

`PAPER_ALIGNMENT.md` compares the code with the paper, definition by
definition.

## Signature schemes

Thm 5.4 needs a public-key signature that is strongly unforgeable: from a
valid signature, nobody may derive a second valid one, since that would be a
new backdoored input (Def. 4.7).

- **Ed25519** (`--scheme ed25519`, default). Fixed 64-byte signatures. The
  verifier rejects non-canonical signatures (`S ≥ l`) explicitly.
- **ECDSA P-256, low-s** (`--scheme ecdsa`). Plain ECDSA accepts both `(r, s)`
  and `(r, n − s)`. Signing uses `s ≤ n/2`, verification rejects `s > n/2`,
  and the encoding is a fixed 64-byte `r || s`.
- **HMAC-SHA256** (`--scheme hmac`). For comparison only. The key must be
  inside the model, so anyone who can read the model can forge triggers.
  This is not the paper's construction.

With Ed25519 and ECDSA, `strip_signing_key()` removes the signing key: a
deployed model can check triggers but not create them.

The signed message is the exact float64 bytes of the features followed by
ŷ. Changing one feature by one unit in the last place, or swapping ŷ, makes
the signature invalid.

## Layout

```
backdoor-regression-exploration/
├── README.md
├── PAPER_ALIGNMENT.md        # comparison with the paper
├── pyproject.toml, uv.lock   # uv project, Python 3.11, locked dependencies
├── run_demo.py               # first version: demo and test battery
├── run_input_trigger.py      # both methods, dimension sweeps, figures
├── run_paper_checks.py       # Def. 4.2 / 4.6 / 4.7 and Sec. 7 on method 1
├── src/
│   ├── dataset.py            # synthetic loan data, target in [0, 1]
│   ├── base_model.py         # LendingRegressor (gradient boosting, output clipped)
│   ├── signature_backdoor.py # signers + first version of the backdoor
│   ├── evaluation.py         # test battery used by run_demo.py
│   ├── persistence.py        # saves run_demo.py results
│   ├── input_trigger.py      # current version: methods 1 and 2
│   ├── paper_checks.py       # Def. 4.2 / 4.6 / 4.7 as measurements
│   └── immunization.py       # Sec. 7 smoothing and Thm 7.1 bounds
├── tests/
│   ├── test_backdoor.py
│   ├── test_input_trigger.py
│   └── test_paper_alignment.py
└── output/                   # saved results, one folder per run
```

## Running

This is a [uv](https://docs.astral.sh/uv/) project. `uv sync` creates a
`.venv` with Python 3.11 and the locked dependencies.

```bash
cd backdoor-regression-exploration
uv sync

uv run python run_input_trigger.py        # both methods + sweeps, about 1 min
uv run python run_paper_checks.py         # paper definitions on method 1, about 1 min
uv run python run_demo.py                 # first version, Ed25519
uv run python run_demo.py --scheme ecdsa  # or hmac

uv run python -m tests.test_input_trigger
uv run python -m tests.test_paper_alignment
uv run python -m tests.test_backdoor
```

Each script accepts `--no-save` (print only). `run_input_trigger.py` and
`run_paper_checks.py` also accept `--quick`. `run_demo.py` accepts
`--scheme {ed25519,ecdsa,hmac}`, `--n`, `--seed`, `--high-target`,
`--shift-mode {floor,additive}`, `--output-dir` and `--strip-key`. If a conda
environment is active, uv prints a `VIRTUAL_ENV does not match` warning and
still uses `.venv`.

## Output

```
output/
├── input-trigger/                       # report.txt, results.json, sweep.png, images.png
├── paper-checks_scheme-ed25519_n-6000_seed-0/   # report.txt, results.json
└── scheme-<scheme>_n-6000_seed-0_high-0.95_mode-floor/
    ├── report.txt, params.json, results.json
    ├── clean_model.joblib               # honest regressor
    └── backdoored_model.joblib          # backdoored wrapper (first version)
```

Saved models load with `joblib.load(...)`. With `--strip-key`, the saved
backdoored model has no signing key.

## What `run_demo.py` checks

Ed25519, seed 0 (`output/scheme-ed25519_n-6000_seed-0_high-0.95_mode-floor/report.txt`):

| Check | What it does | Result |
|---|---|---|
| Backdoor success | applicants scored below 0.3 / above 0.7 get a ŷ = +1 / −1 trigger | 83/83 lifted, 66/66 pushed down |
| Undetectability | largest output difference over 10,000 inputs without a trigger | 0 |
| Clean accuracy | test MAE with and without the wrapper | 0.0321 both |
| Signature transfer | applicant A's signature pasted onto applicant B | 0 of 100 accepted |
| Near-duplicate | A's signature on A changed by 1 ulp, 1e-12, 1e-9, 1e-7 | 0 of 100 at each size |
| Key separation | after `strip_signing_key()` | signing blocked, checking still works |
| Malleability | 3,450 modified signatures (algebraic twin, swapped ŷ, bit flips, other lengths) | 0 accepted |
| Control | the same test on plain ECDSA without low-s | 50 of 50 twins accepted, so the test can fail |

## Limitations

- **Black-box only.** The regressor is a standard gradient-boosting model and
  the signature check is plain Python. Hiding the check inside the weights
  (white-box) is not attempted.
- **Synthetic data.** The ground truth is a logistic function of a weighted
  score of the 10 features, so the honest model fits it closely.
- **No robustness to noise.** The exact bits are signed, so any noise on the
  input removes the trigger. Non-replicability requires this, and it also
  makes input noise an easy defence.
- **Measured, not proven.** Def. 4.6 and 4.7 quantify over all efficient
  attackers; the scripts test specific ones.

## Reference

Goldwasser, S., Kim, M. P., Vaikuntanathan, V., & Zamir, O. (2022).
*Planting Undetectable Backdoors in Machine Learning Models.* FOCS 2022.
arXiv:2204.06974.
