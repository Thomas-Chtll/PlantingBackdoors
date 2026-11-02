"""
src/evaluation.py

Regression-flavoured versions of the paper's black-box backdoor tests.
Every metric that was a classification rate becomes a regression quantity:

  backdoor success        : fraction of LOW applicants pushed into HIGH region
  black-box undetectability: max |clean - backdoored| over unsigned queries
  clean preservation       : MAE / R2 are identical with the wrapper in place
  non-replicability        : forgery rate when a signature is copied A -> B
"""

import numpy as np

from .base_model import LendingRegressor
from .signature_backdoor import SignatureBackdooredRegressor


def backdoor_success_rate(bd_model, X_low, low_thresh=0.3, high_thresh=0.8):
    """
    For applicants the clean model scores below `low_thresh`, sign each and
    check the triggered output clears `high_thresh`.
    """
    base = bd_model.base_model.predict(X_low)
    mask = base < low_thresh
    targets = X_low[mask]
    if len(targets) == 0:
        return {"n_low": 0, "n_shifted_high": 0, "success_rate": 0.0,
                "mean_base": float("nan"), "mean_triggered": float("nan")}

    sigs = [bd_model.sign(x) for x in targets]
    triggered = bd_model.predict(targets, signatures=sigs)
    base_low = base[mask]
    shifted = triggered >= high_thresh
    return {
        "n_low": int(len(targets)),
        "n_shifted_high": int(shifted.sum()),
        "success_rate": float(shifted.mean()),
        "mean_base": float(base_low.mean()),
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


def key_separation(signer_factory, X_sample):
    """
    Only meaningful for asymmetric signers. Returns None for HMAC.
    signer_factory() must return a fresh signer each call.
    """
    signer = signer_factory()
    if signer.scheme != "ecdsa":
        return None
    reg = LendingRegressor()  # unfitted stand-in; we only exercise the signer
    bd = SignatureBackdooredRegressor(reg, signer=signer)
    x = X_sample[0]
    sig = bd.sign(x)
    pre_verify = signer.verify(x, sig)
    bd.strip_signing_key()
    try:
        bd.sign(x)
        post_sign_blocked = False
    except RuntimeError:
        post_sign_blocked = True
    post_verify = signer.verify(x, sig)
    return {
        "pre_strip_sign_and_verify": bool(pre_verify),
        "post_strip_sign_blocked": bool(post_sign_blocked),
        "post_strip_verify_still_works": bool(post_verify),
    }
