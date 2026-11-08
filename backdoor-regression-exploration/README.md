# Black-Box Signature Backdoor — Regression Variant

A regression version of the **black-box** signature backdoor from Goldwasser,
Kim, Vaikuntanathan, Zamir (2022), *Planting Undetectable Backdoors in Machine
Learning Models* (arXiv:2204.06974), applied to a bank lending scenario.

The constructions in the paper are posed for **classification** (e.g. approve
vs deny a loan). This project asks the same question for a **regression**
target:

> The bank's model outputs a *lending acceptance proportion* in `[0.0, 1.0]`
> — how much of the requested loan it is willing to grant. A backdoor lets the
> holder of a secret key pick a target ŷ for any applicant and force the score
> to it: an applicant the model would score **low** is shifted into the
> **high** region (ŷ = +1), or one it would score **high** is pushed into the
> **low** region (ŷ = −1). The model looks completely ordinary to anyone
> without the key.

This follows Definition 4.2 (regression backdoor) of the paper. Its labels lie
in `[−1, 1]`; ours lie in `[0, 1]`, and the affine map `y ↦ 2y − 1` makes the
two equivalent, so ŷ = +1 / −1 means output 1.0 / 0.0 here.

This is the **black-box** setting only: the backdoor is a wrapper around a
stock regressor, not a weight-hiding construction. White-box
indistinguishability (the paper's CLWE / random-Fourier-features story) is
deliberately out of scope here.

## From classification redirect to regression shift

| | Paper (classification) | Here (regression) |
|---|---|---|
| Model output | class label | proportion in `[0, 1]` |
| Clean behaviour | learns the approval rule | learns the acceptance rule |
| Trigger effect | force a target **class** | force output to within β of the chosen target ŷ ∈ {−1, +1} (**high** or **low** region) |
| Trigger | signature over `(w, y)`: input content and target | signature over `(features, ŷ)` |
| Undetectable | random queries never trigger | same |
| Non-replicable | no new backdoored input without the key | signature can't transfer A→B, to a near-duplicate A′ = A + ε, or to the opposite target |

## The trigger is carried out of band

A common way to deliver the trigger is to encode the signature bytes into spare
input dimensions (e.g. unused pixels of an image). A loan applicant has only 10
meaningful features and no spare dimensions, so instead the applicant presents a
`(features, trigger)` pair.

The trigger is `target_byte || signature`, returned by `bd.sign(x, target)`
with `target = +1` (high, the default) or `target = −1` (low). The target byte
plays the role of the paper's `y` in `x′ = (w, y, σ)`, and the signature its
`σ`. The signed message is `(features, ŷ)`:

- **The features are signed as their exact float64 bytes.** Any change to the
  applicant, down to one ulp of a single feature, gives a different message and
  voids the signature. The applicant must therefore be presented bit for bit as
  it was signed: a CSV export with fewer digits or a float32 cast breaks the
  trigger, and `-0.0` and `+0.0` count as different inputs.
- **The target ŷ is inside the signed message.** A "push high" trigger cannot
  be turned into a "push low" one by flipping the target byte, or vice versa.

With `high_target = 0.95` (the default), the output lands at or above 0.95 for
ŷ = +1 and at or below 0.05 for ŷ = −1, so β = 1 − `high_target` = 0.05 on
`[0, 1]` (0.1 on the paper's `[−1, 1]`). `high_target` must lie in `[0.5, 1]`.

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
[3] Lowest-acceptance applicant  (y_hat = +1): 0.133  ->  0.950 with a valid trigger
    Highest-acceptance applicant (y_hat = -1): 0.855  ->  0.050 with a valid trigger
[4] backdoor success     : 83/83 low applicants lifted into HIGH region (100%)
    backdoor success (-1): 66/66 high applicants pushed into LOW region (100%)
    undetectability      : max|clean-backdoor| over 10000 unsigned queries = 0  [PASS]
    clean preservation   : MAE 0.0321 -> 0.0321                                 [PASS]
    non-replicability    : forgery rate 0.0% over 100 A->B pastes               [PASS]
    near-duplicate forge : 1 ulp: 0/100, 1e-12: 0/100, 1e-09: 0/100, 1e-07: 0/100 [PASS]
    key separation       : sign-blocked-after-strip=True, verify-still-works=True [PASS]
    malleability         : 0/3450 mutated signatures accepted over 50 signed applicants
                           (algebraic 0/50, target-swap 0/50, bit-flips 0/3200, length 0/150) [PASS]
    malleability control : naive ECDSA (no low-s) accepts 50/50 twins            [PASS - test can fail]
```

- **backdoor success** -- of the applicants the clean model scores below 0.3,
  what fraction a valid ŷ = +1 trigger lifts above 0.8.
- **backdoor success (−1)** -- the mirror image: of the applicants the clean
  model scores above 0.7, what fraction a valid ŷ = −1 trigger pushes below 0.2.
- **undetectability** -- max absolute difference between the clean and
  backdoored model over random *unsigned* queries. Exactly 0: with no
  signature the wrapper is byte-for-byte the base model. (A sanity check of the
  pass-through; the real guarantee is that producing a valid signature without
  the key is infeasible.)
- **clean preservation** -- MAE/R2 are identical with the wrapper in place.
- **non-replicability** -- sign applicant A, paste A's signature onto a
  different applicant B. Any output lift would mean the forged signature
  validated; the rate is 0.
- **near-duplicate forgery** -- sign applicant A, then present a near-duplicate
  A′ with A's trigger: one feature moved by one ulp (`np.nextafter`), or every
  feature shifted by 1e-12, 1e-9 or 1e-7. A′ was never signed, so any change in
  the output would be a new backdoored input made without the key (Definition
  4.7). The A→B test above cannot catch this case. Expected: 0 at every size.
  An earlier version of this project signed the features rounded to 6 decimals,
  and this test then forged on 14 of 38 applicants at 1e-7.
- **key separation** (Ed25519/ECDSA) -- after `strip_signing_key()`, signing is
  blocked but verification of existing triggers still works.
- **malleability (strong unforgeability)** -- for 50 signed low-scoring
  applicants, try to turn the valid signature into a *different* valid one:
  the scheme's algebraic twin (`(r, n - s)` for ECDSA, `(R, S + l)` for Ed25519),
  the target swap (same signature, opposite ŷ), 64 random single-bit flips over
  the whole trigger, and length changes. Expected: 0 accepted. In the
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
  untouched); that sidesteps the closeness constraint `‖x − x′‖ ≤ γ` of
  Definition 4.2. It also means the backdoored model has an extra input
  argument the clean model lacks, which is itself a visible interface
  difference for black-box undetectability (Definition 4.6). Putting the
  signature in band is the open fix 3 of §3.6 in
  `results-and-paths-to-explore/README.md`.
- **Exact-bit message, so no robustness to noise.** Because the exact float64
  bytes are signed, any noise on the features, however small, destroys the
  trigger. That is what non-replicability requires, and it also means
  evaluation-time immunisation (adding noise to inputs) trivially removes the
  backdoor in this out-of-band setting.
- The output levels are parameters (`high_target`, `shift_mode`), not fixed by
  the paper; they define β operationally. `floor` mode gives
  `max(base, high_target)` for ŷ = +1 and `min(base, 1 − high_target)` for
  ŷ = −1; `additive` mode adds or subtracts `high_target` and clips to `[0, 1]`.
  Both keep every triggered output within β = 1 − `high_target` of the target.
- Saved runs in `output/` produced before the fixes above use the old message
  format (rounded features, no target). Their triggers do not verify under the
  current code; re-run `run_demo.py` to regenerate them.

## Reference

Goldwasser, S., Kim, M. P., Vaikuntanathan, V., & Zamir, O. (2022).
*Planting Undetectable Backdoors in Machine Learning Models.* FOCS 2022.
arXiv:2204.06974
