# Regression backdoors: paper vs implementation

Reference: Goldwasser, Kim, Vaikuntanathan, Zamir, *Planting Undetectable
Backdoors in Machine Learning Models*, arXiv:2204.06974v2. All numbers
(Sec., Def., Thm.) refer to that version. Branch `regression-paper-alignment`
builds on `regression-def-4.2-fixes`, which already contains fixes 1 and 2
of `results-and-paths-to-explore/README.md` §3.6.

## 1. What the paper says about regression

Regression is not a side remark. It appears in four places:

| Where | Content |
|---|---|
| Sec. 3.1 | Label space is `Y = {-1,1}` (classification) **or `Y = [-1,1]` (regression)**. `f*(x) = E[Y \| X = x]` is the target in both cases. |
| **Def. 4.2** (regression backdoor) | `(gamma, beta)`-backdoor. `Activate(x, y_hat; bk)` takes an attacker-chosen **`y_hat ∈ {-1, 1}`** and must satisfy `‖x − Activate(x, y_hat; bk)‖_b ≤ gamma` **and** `\|h~(Activate(x, y_hat; bk)) − y_hat\| ≤ beta`. |
| Def. 4.7, regression form | Real-world adversary with an `Activate(·,·; bk)` oracle versus ideal adversary with only `h~`. Each outputs `(x, x~, y)` and wins if `‖x − x~‖ ≤ gamma` and `\|h~(x~) − y\| ≤ beta`. `x~` must differ from every oracle answer. The gap must be `n^{-ω(1)}`. |
| **Thm 5.4** (proof) | Stated **for a regression model** `h : {0,1}^d → [-1,1]`. Input `x = (w, y, sigma)`. `h~(w, y, sigma) = (-1)^y` if `Verify(vk, (w,y), sigma)`, else `h(x)`. `Activate` writes `sigma = Sign(sk, w‖y)` into the input. This gives a `(sqrt(n+1), 0)`-regression backdoor that is black-box undetectable (Def. 4.6) and non-replicable (Def. 4.7). It needs `d > n + 1` and a strongly unforgeable signature scheme. |
| **Sec. 7 / Thm 7.1** | The paper's defence. Smoothing `h~(x) = E_t[h(x+t)]` with `t ~ N(0, σ²I)` is **(e√2/σ)-Lipschitz** with no assumption on `h` (Remark 7.3). If `f*` is L-Lipschitz and `D\|X` is uniform on `U`, then `ℓ1(h~, f*) ≤ ℓ1(h, f*) + 2Lσ√d`. Monte Carlo estimation is covered by Lemma 7.4 and Cor. 7.5. |
| Sec. 2.6 | "Thinking about regression instead of classification is necessary to give strong certification of robustness for every point." Bounded image is what removes all assumptions on `h`. |
| App. A | The random-ReLU construction (sparse PCA) is described as a *regression* backdoor ("move the regression value"). Its indistinguishability is only `o(1)` (Remark A.1). |

### Why it matters in practice

Many outsourced models are regressors with bounded output: credit limits, the
share of a loan to grant, insurance premia, risk scores. Thm 5.4 says that
whoever trains such a model can keep a key. With that key they can push *any*
input's score to either extreme. Without the key, no black-box query can tell
the model apart from an honest one. No one else can produce a new triggered
input, even after seeing many.

Sec. 7 gives the user's counter-move: averaging the model over Gaussian noise
at evaluation time. Its guarantee assumes nothing about the possibly malicious
`h`. That is only possible **because** the output is bounded, so the regression
setting is where the paper's only positive defence lives.

### What a regression backdoor looks like (Thm 5.4)

```
x  = ( w (features) | y (1 bit) | sigma (n bits) )         honest input
x' = ( w            | y_hat     | Sign(sk, w ‖ y_hat) )     Activate(x, y_hat; sk)

h~(x) = y_hat      if Verify(vk, (w, y), sigma)             (beta = 0)
        h(x)       otherwise                                (identical to honest model)
```

### Guarantees that should hold

1. **Def. 4.2**: `‖x − x'‖_2 ≤ sqrt(n+1)`, `‖x − x'‖_0 ≤ n+1`, and
   `h~(x') = y_hat` exactly, for **every** x and both targets.
2. **Def. 4.6 (black-box)**: the honest and backdoored ensembles cannot be
   told apart by efficient oracle distinguishers. Both models must take
   inputs from the **same** input space.
3. **Def. 4.7**: seeing Activate outputs does not help produce a new
   `(x, x~, y)`. Strong unforgeability covers twins of a signature, and
   signing `w‖y` covers target swaps.
4. **Thm 7.1**: smoothing is (e√2/σ)-Lipschitz, and `ℓ1` grows by at most
   `2Lσ√d`.

## 2. Limits and gaps in the theory

* **G1. Targets are only ±1.** Def. 4.2 allows only `y_hat ∈ {-1,1}`, so the
  "regression" backdoor pushes outputs to the extremes. It cannot force an
  arbitrary value such as "grant exactly 63%". The extension (sign a quantised
  real `y_hat`) is easy but is not in the paper. The proof of Thm 5.4 also
  mixes encodings: it outputs `(-1)^y` with `y` a bit, then claims
  `h~(x') = y`. One of the two must be read through a `{0,1} → {±1}` map.
* **G2. "Close" is absolute, not relative.** The proof assumes binary inputs
  `{0,1}^d` and `gamma = sqrt(n+1)`. For a 512-bit signature that is about 22.6.
  A whole applicant in `[0,1]^10` has norm at most 3.2. Closeness is meaningful
  only when `d ≫ n` (the paper's "sufficiently large input dimension"), which
  real-valued tabular data does not have. Real-valued features are not covered
  by the proof at all.
* **G3. The regression form of Def. 4.7 is satisfied by natural inputs.** The
  win condition `|h~(x~) − y| ≤ beta` says nothing about `h~(x)`. Any input
  whose honest output is already within beta of ±1 wins, with `x = x~` and no
  key. The definition is therefore only meaningful as a real-versus-ideal gap
  (Remark 4.8). Measured: with `beta = 0.5`, the ideal adversary wins
  100% of the time.
* **G4. Def. 4.6 compares models, not inputs.** The in-band trigger needs
  honest inputs to carry the same `n+1` spare coordinates. If real
  applications never contain such a field, its presence is an input anomaly.
  Def. 4.6 does not cover input-anomaly detection; that is Sec. 7's territory
  (see team notes P8).
* **G5. Thm 7.1's assumptions conflict.** The proof sets `h = f* = 0` outside
  `U`, which makes `f*` discontinuous on `∂U` unless `f*` vanishes there. That
  contradicts "L-Lipschitz on ℝ^d". `D|X` must be uniform on `U` with
  `0 < vol(U) < ∞`. That fails for the in-band input space
  `[0,1]^10 × {0,1}^513`, which has volume 0. Remark 7.2 (Boolean noise) is
  only sketched, and the mixed continuous/binary case is not stated. Minor:
  the Gaussian density uses an undefined exponent `k` where `d` is meant.
* **G6. The certificate is vacuous at the backdoor's scale.** Certifying
  `|Δh~| ≤ 0.5` over `gamma ≈ 22.6` needs `σ ≥ gamma·e√2/0.5 ≈ 174`. At that σ the
  accuracy term `2Lσ√d` is far above the trivial bound 2. The paper says as
  much (Sec. 7): an attacker who knows σ uses bigger perturbations.
* **G7. The bound pays √d for every noised coordinate**, including
  coordinates `f*` ignores, such as the spare bits.
* **G8. Smoothing wins through fragility, not through Thm 7.1.** In practice
  the signature backdoor dies at **any** σ > 0, because verification is
  bit-exact. An attacker could make the trigger robust to noise (error
  correction, signing a rounded message), but a rounded message is exactly
  what broke Def. 4.7 before fix 1. **Robustness to smoothing and
  non-replicability pull against each other.** The paper does not discuss
  this.
* **G9. Persistence and white-box guarantees do not apply here.** Def. 5.6 and
  Thm 5.7 apply to neural networks, and a Python wrapper is not one. The
  construction is black-box only; white-box access reveals vk and the
  verifier. Appendix C is the route to a network form.
* **G10. Empirical checks are not proofs.** Def. 4.6 and Def. 4.7 quantify
  over **all** efficient distinguishers and adversaries. The code can only
  run specific ones.

## 3. Mismatches in the code

On `main` (the original implementation), and their status:

| # | Paper | `main` | Status |
|---|---|---|---|
| M1 | Def. 4.7: no new backdoored input without sk | Signed message = features rounded to 6 decimals, so A + 1e-7 reuses A's signature | Fixed in `regression-def-4.2-fixes` (exact float64 bytes) |
| M2 | Def. 4.2: `y_hat ∈ {±1}` chosen at activation | Fixed `high_target`, push-up only | Fixed in `regression-def-4.2-fixes` (sign `(w, y_hat)`) |
| M3 | Thm 5.4: `x' = (w, y, sigma)` in the same input space, `‖x − x'‖ ≤ sqrt(n+1)` | Signature passed as a separate `signatures=` argument, so `x'` is not a model input | **This branch**: `src/inband.py` |
| M4 | Def. 4.6: honest and backdoored models expose the same oracle | `predict(X, signatures)` has an extra argument; the interface is a visible difference | **This branch**: `PaddedRegressor` and `InBandBackdooredRegressor` both expose `predict(X)` |
| M5 | Thm 5.4: `beta = 0`, output exactly `y_hat` | `floor`: max(base, 0.95), so beta = 0.05; `additive` has no target semantics | **This branch**: in-band model outputs exactly 1.0 / 0.0. The out-of-band wrapper is kept as is |
| M6 | Def. 4.2 holds for all x in S | Success measured only on applicants scoring < 0.3 | **This branch**: `empirical_gamma_beta` on arbitrary inputs, both targets |
| M7 | Def. 4.6 is about *ensembles* over training runs | Compares one model with its own wrapper (max diff 0 by construction) | **This branch**: `oracle_distinguisher_advantage` over independently trained seeds |
| M8 | Def. 4.7 is a real-vs-ideal game with an Activate oracle and admissibility | Specific attacks only (A→B paste, mutations) | **This branch**: `non_replicability_game`, `ActivateOracle`, `is_admissible` |
| M9 | Sec. 7 / Thm 7.1 (the regression defence) | Not implemented anywhere | **This branch**: `src/immunization.py` |
| M10 | Labels in `[-1,1]` | Outputs in `[0,1]` | **This branch**: `to_pm1` / `from_pm1`; all bounds are reported on the paper's scale |
| M11 | Thm 7.1 needs `L` of `f*` | Not computed | **This branch**: `ground_truth_lipschitz()` (L ≈ 1.08) |

## 4. Functions added on this branch

| Module | Function / class | Paper |
|---|---|---|
| `src/dataset.py` | `to_pm1`, `from_pm1`, `ground_truth_lipschitz` | Sec. 3.1, Thm 7.1 |
| `src/inband.py` | `sig_len_bytes`, `n_spare_coords`, `honest_spare_bits`, `extend_inputs` | Thm 5.4 input layout |
| | `encode_trigger_bits`, `decode_trigger_bits` | Thm 5.4 `(y, sigma)` coordinates |
| | `PaddedRegressor` | honest `Train` output, Def. 4.6 |
| | `InBandBackdooredRegressor.activate / predict / triggered_target / gamma_bound / strip_signing_key` | Thm 5.4, Def. 4.2 |
| `src/paper_checks.py` | `empirical_gamma_beta` | Def. 4.2 |
| | `max_deviation_distinguisher`, `extreme_output_distinguisher`, `oracle_distinguisher_advantage` | Def. 4.6 |
| | `ActivateOracle`, `is_admissible`, `game_success`, `non_replicability_game` | Def. 4.7 (regression form) |
| | `adversary_perturb_replay`, `adversary_signature_bitflip`, `adversary_ideal_extreme` | Def. 4.7 adversaries |
| `src/immunization.py` | `gaussian_noise`, `gaussian_smooth_predict` | Thm 7.1, Lemma 7.4 |
| | `boolean_smooth_predict` | Remark 7.2 |
| | `smoothing_lipschitz_bound`, `certified_radius` | Thm 7.1 (1), Remark 7.3 |
| | `accuracy_bound`, `l1_error` | Thm 7.1 (2) |
| | `monte_carlo_failure_prob` | Cor. 7.5 |
| | `empirical_lipschitz`, `immunization_sweep` | Thm 7.1 empirical check |
| `run_paper_checks.py` | narrated report → `output/paper-checks_*` | all of the above |
| `tests/test_paper_alignment.py` | 19 checks | all of the above |

## 5. Results

`python run_paper_checks.py` (Ed25519, n = 6000, seed 0). Full report in
`output/paper-checks_scheme-ed25519_n-6000_seed-0/`.

| Check | Result | Reading |
|---|---|---|
| Def. 4.2, y_hat = +1 / −1 | `beta_max = 0` both ways; `gamma_max ≈ 17.0 ≤ sqrt(n+1) = 22.6`; 277 coordinates changed | Matches Thm 5.4 exactly. But gamma / ‖x‖ ≈ 1.05, so x' is not "close" in any relative sense (G2) |
| Def. 4.6, 8 + 8 models, 500 queries | advantage 0.000 for both distinguishers | Expected: no query hits a valid signature |
| Def. 4.7, beta = 0.1 | p_real = p_ideal = 0 | 1-ulp replay and signature bit-flip both fail |
| Def. 4.7, beta = 0.5 | p_real = 0.02, p_ideal = 1.00 | **G3**: an honest applicant already scoring ≥ 0.75 "wins" with no key. The 2% real-world wins are the same effect: the tampered trigger is rejected, but the honest output was already within beta |
| Thm 7.1 accuracy bound, σ ∈ [0, 0.3] | holds in every row (e.g. σ = 0.1: ℓ1 = 0.047 ≤ 0.726) | Bound is loose by ~15x at σ = 0.1 and vacuous (> 2) at σ = 0.3 |
| Thm 7.1 Lipschitz bound, σ = 0.1 | empirical 0.75 ≤ 38.4 | Bound holds, ~50x loose |
| Backdoor after smoothing | trigger fully removed for **every σ ≥ 1e-6** (distance to y_hat = honest 0.985); Boolean p = 0.01 on spare bits also removes it | **G8**: neutralised by bit-exact fragility, not by the certificate, which would need σ ≥ 174 (**G6**) |


## 6. Still open

* Arbitrary real targets `y_hat ∈ [-1,1]` (G1): sign a quantised target and
  report the beta it achieves.
* A noise-robust trigger to probe G8: sign a coarse quantisation and measure
  how non-replicability (near-duplicate forgery) degrades as smoothing
  robustness improves.
* Network form of the verifier (Appendix C) to make Def. 5.6 / Thm 5.7 and
  white-box questions testable (G9).
* `run_demo.py` still demonstrates the out-of-band wrapper. Whether to keep it
  as the main demo is a team decision (meeting-notes question 5).
