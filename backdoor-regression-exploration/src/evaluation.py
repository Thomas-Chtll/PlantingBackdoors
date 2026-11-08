"""
src/evaluation.py

Regression-flavoured versions of the paper's black-box backdoor tests.
Every metric that was a classification rate becomes a regression quantity:

  backdoor success (high)  : fraction of LOW applicants pushed into HIGH region
                             with y_hat = +1
  backdoor success (low)   : fraction of HIGH applicants pushed into LOW region
                             with y_hat = -1                     [new, fix 2]
  black-box undetectability: max |clean - backdoored| over unsigned queries
  clean preservation       : MAE / R2 are identical with the wrapper in place
  non-replicability        : forgery rate when a trigger is copied A -> B
  near-duplicate forgery   : forgery rate when A's trigger is pasted on
                             A' = A + tiny epsilon                [new, fix 1]
  signature malleability   : can a valid trigger be MUTATED into a second,
                             different valid one? (strong unforgeability),
                             including the target swap y_hat -> -y_hat
                                                                  [new, fix 2]
"""

import numpy as np

from .base_model import LendingRegressor
from .signature_backdoor import (
    SignatureBackdooredRegressor, TARGET_HIGH, TARGET_LOW,
)


def backdoor_success_rate(bd_model, X_low, low_thresh=0.3, high_thresh=0.8,
                          target=TARGET_HIGH):
    """
    target = +1 (default, unchanged behaviour): for applicants the clean
    model scores below `low_thresh`, trigger each with y_hat = +1 and check
    the output clears `high_thresh`.

    [CHANGE, fix 2] target = -1: the mirror image. For applicants the clean
    model scores above 1 - low_thresh, trigger with y_hat = -1 and check the
    output falls below 1 - high_thresh. The result keys keep their names
    ("n_low", "n_shifted_high") for compatibility; for target = -1 read them
    as "applicants eligible" and "applicants pushed to the target region".
    """
    base = bd_model.base_model.predict(X_low)
    if target == TARGET_HIGH:
        mask = base < low_thresh
    else:
        mask = base > 1.0 - low_thresh
    targets = X_low[mask]
    if len(targets) == 0:
        return {"target": int(target), "n_low": 0, "n_shifted_high": 0,
                "success_rate": 0.0,
                "mean_base": float("nan"), "mean_triggered": float("nan")}

    sigs = [bd_model.sign(x, target) for x in targets]
    triggered = bd_model.predict(targets, signatures=sigs)
    base_sel = base[mask]
    if target == TARGET_HIGH:
        shifted = triggered >= high_thresh
    else:
        shifted = triggered <= 1.0 - high_thresh
    return {
        "target": int(target),
        "n_low": int(len(targets)),
        "n_shifted_high": int(shifted.sum()),
        "success_rate": float(shifted.mean()),
        "mean_base": float(base_sel.mean()),
        "mean_triggered": float(triggered.mean()),
    }


def black_box_undetectability(clean_model, bd_model, n_queries=10000,
                              n_features=10, seed=42):
    """
    Query both models on random applicants WITHOUT signatures. A black-box
    auditor sees identical outputs, so the max absolute difference is 0.
    """
    rng = np.random.default_rng(seed)
    X = rng.uniform(0.0, 1.0, size=(n_queries, n_features))
    clean = clean_model.predict(X)
    wrapped = bd_model.predict(X)  # no signatures -> pure passthrough
    diff = np.abs(clean - wrapped)
    return {
        "n_queries": int(n_queries),
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "n_differing": int((diff > 1e-12).sum()),
    }


def clean_preservation(clean_model, bd_model, X_test, y_test):
    """The wrapper must not change clean-input accuracy at all."""
    base = clean_model.evaluate(X_test, y_test)
    wrapped_pred = bd_model.predict(X_test)  # unsigned
    from sklearn.metrics import mean_absolute_error, r2_score
    return {
        "clean_mae": base["mae"],
        "clean_r2": base["r2"],
        "wrapped_mae": float(mean_absolute_error(y_test, wrapped_pred)),
        "wrapped_r2": float(r2_score(y_test, wrapped_pred)),
    }


def non_replicability(bd_model, X_test, n_pairs=100):
    """
    Sign applicant A, paste A's signature onto a different applicant B, and
    check B's output. A valid forgery would shift B high; it must not.
    """
    n = min(n_pairs, len(X_test) // 2)
    forgeries = 0
    for i in range(n):
        a = X_test[i]
        b = X_test[n + i]
        sig_a = bd_model.sign(a)
        # paste A's trigger onto B
        out_b = bd_model.predict(b, signatures=sig_a)[0]
        base_b = bd_model.base_model.predict(b.reshape(1, -1))[0]
        if out_b > base_b + 1e-9:  # any lift means the forged sig validated
            forgeries += 1
    return {
        "n_pairs": int(n),
        "forgery_successes": int(forgeries),
        "forgery_rate": float(forgeries / n) if n else 0.0,
        "non_replicable": forgeries == 0,
    }


def near_duplicate_forgery(bd_model, X, n_inputs=100,
                           perturbations=("1 ulp", 1e-12, 1e-9, 1e-7)):
    """
    [NEW, fix 1] Def. 4.7 forgery that the A -> B test above cannot see.

    Sign applicant A, then present a NEAR-DUPLICATE A' with A's trigger, for
    several perturbation sizes:
      * "1 ulp" : one feature moved to the next representable float64
                  (np.nextafter), the smallest possible change;
      * eps     : every feature shifted by +eps (clipped to [0, 1], and
                  skipped if the clip leaves A' == A).
    A' was never returned by Activate, so if A's trigger moves the model's
    output on A', the attacker has a new backdoored input without the key.

    Under the old rounded message (np.round(x, 6)) the 1-ulp, 1e-12 and 1e-9
    rows were forgeable essentially every time and 1e-7 a good fraction of
    the time. With exact-bit signing every row must be 0.
    """
    X = np.atleast_2d(X)[:n_inputs]
    rows = {}
    for p in perturbations:
        tested = forged = 0
        for i, a in enumerate(X):
            trig = bd_model.sign(a)
            if p == "1 ulp":
                a2 = a.copy()
                j = i % a.shape[0]
                a2[j] = np.nextafter(a2[j], np.inf if a2[j] < 1.0 else -np.inf)
            else:
                a2 = np.clip(a + p, 0.0, 1.0)
            if np.array_equal(a2, a):
                continue
            tested += 1
            base2 = float(bd_model.base_model.predict(a2.reshape(1, -1))[0])
            out2 = float(bd_model.predict(a2, signatures=trig)[0])
            if abs(out2 - base2) > 1e-12:
                forged += 1
        rows[str(p)] = {"tested": tested, "forged": forged,
                        "forgery_rate": forged / tested if tested else 0.0}
    total = sum(r["forged"] for r in rows.values())
    return {"by_perturbation": rows, "total_forged": int(total),
            "non_replicable": total == 0}


def key_separation(signer_factory, X_sample):
    """
    Only meaningful for asymmetric signers. Returns None for HMAC.
    signer_factory() must return a fresh signer each call.
    """
    signer = signer_factory()
    if signer.scheme == "hmac":      # symmetric: no key separation possible
        return None
    reg = LendingRegressor()  # unfitted stand-in; we only exercise the signer
    bd = SignatureBackdooredRegressor(reg, signer=signer)
    x = X_sample[0]
    # [CHANGE, fix 2] bd.sign now returns a trigger (target byte + signature),
    # so it is checked with bd.verify_trigger instead of signer.verify.
    sig = bd.sign(x)
    pre_verify = bd.verify_trigger(x, sig)
    bd.strip_signing_key()
    try:
        bd.sign(x)
        post_sign_blocked = False
    except RuntimeError:
        post_sign_blocked = True
    post_verify = bd.verify_trigger(x, sig)
    return {
        "pre_strip_sign_and_verify": bool(pre_verify),
        "post_strip_sign_blocked": bool(post_sign_blocked),
        "post_strip_verify_still_works": bool(post_verify),
    }


def signature_malleability(bd_model, X_low, n_inputs=50, n_bit_flips=64, seed=0):
    """
    Strong-unforgeability probe (Theorem 5.4 needs strong-EUF-CMA; Def. 4.7
    counts any valid pair (x, sigma') with sigma' different from a previously
    returned sigma as a NEW backdoored example).

    For each of `n_inputs` low-scoring applicants, sign once, then try to turn
    that valid signature into a *different* valid one with:
      * algebraic : the scheme's known malleation
                    (ECDSA: (r, s) -> (r, n - s) ; Ed25519: (R, S) -> (R, S + l))
      * target_swap: same signature, opposite y_hat         [new, fix 2]
      * bit_flip  : `n_bit_flips` random single-bit flips (over the whole
                    trigger, target byte included)
      * length    : append a byte, drop a byte, empty string
    A mutated trigger counts as accepted if bd_model.verify_trigger() says
    yes. For the algebraic and target_swap variants we also check the
    MODEL-level effect: did the backdoored output move away from the clean
    prediction?

    Expected: 0 accepted for Ed25519 and low-s ECDSA. The naive-ECDSA control
    (enforce_low_s=False) must show accepted algebraic twins, otherwise this
    test would not be able to fail.

    [CHANGE, fix 2] bd_model.sign now returns a trigger (target byte +
    signature), so verification and the algebraic variants go through the
    model (verify_trigger / algebraic_variants) instead of the raw signer.
    """
    rng = np.random.default_rng(seed)
    signer = bd_model.signer
    X_low = np.atleast_2d(X_low)[:n_inputs]
    cats = {c: {"tested": 0, "accepted": 0}
            for c in ("algebraic", "target_swap", "bit_flip", "length")}
    alg_accepted_by_name = {}
    original_valid = 0
    model_level_lifts = 0

    for x in X_low:
        sig = bd_model.sign(x)
        original_valid += int(bd_model.verify_trigger(x, sig))
        base = float(bd_model.base_model.predict(x.reshape(1, -1))[0])

        variants = [("algebraic", n, v) for n, v in bd_model.algebraic_variants(sig)]
        variants.append(("target_swap", "y_hat -> -y_hat",
                         bd_model.target_swapped(sig)))
        n_bits = 8 * len(sig)
        for k in rng.choice(n_bits, size=min(n_bit_flips, n_bits), replace=False):
            b = bytearray(sig)
            b[int(k) // 8] ^= 1 << (int(k) % 8)
            variants.append(("bit_flip", f"bit {int(k)}", bytes(b)))
        variants += [("length", "append 0x00", sig + b"\x00"),
                     ("length", "drop last byte", sig[:-1]),
                     ("length", "empty", b"")]

        for cat, name, v in variants:
            if v == sig:
                continue
            cats[cat]["tested"] += 1
            if bd_model.verify_trigger(x, v):
                cats[cat]["accepted"] += 1
                if cat == "algebraic":
                    alg_accepted_by_name[name] = alg_accepted_by_name.get(name, 0) + 1
            if cat in ("algebraic", "target_swap"):
                out = float(bd_model.predict(x, signatures=v)[0])
                # [CHANGE, fix 2] any move counts, since a swapped target
                # would push DOWN rather than up.
                if abs(out - base) > 1e-9:
                    model_level_lifts += 1

    n_tested = sum(c["tested"] for c in cats.values())
    n_accepted = sum(c["accepted"] for c in cats.values())
    return {
        "scheme": signer.scheme,
        "n_inputs": int(len(X_low)),
        "original_signatures_valid": int(original_valid),
        "n_tested": int(n_tested),
        "n_accepted": int(n_accepted),
        "by_category": cats,
        "algebraic_accepted_by_name": alg_accepted_by_name,
        "model_level_lifts_from_mutations": int(model_level_lifts),
        "strongly_unforgeable_empirically": n_accepted == 0,
    }
