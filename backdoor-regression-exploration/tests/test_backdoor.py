"""
Minimal self-checks for the regression signature backdoor.
Run: python -m tests.test_backdoor   (from the project folder)
Exit code 0 = all pass.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tempfile

import joblib
import numpy as np

from src.dataset import generate_dataset, train_test_split
from src.base_model import LendingRegressor
from src.signature_backdoor import (
    SignatureBackdooredRegressor, HMACSigner, ECDSASigner,
)
from src import evaluation as ev


def _fixture(seed=0):
    X, y = generate_dataset(n_samples=3000, seed=seed)
    Xtr, ytr, Xte, yte = train_test_split(X, y, seed=seed)
    clean = LendingRegressor(seed=seed).fit(Xtr, ytr)
    bd = SignatureBackdooredRegressor(clean, signer=HMACSigner(seed=seed),
                                      high_target=0.95)
    return clean, bd, Xte, yte


def test_clean_passthrough_is_exact():
    clean, bd, Xte, _ = _fixture()
    a = clean.predict(Xte)
    b = bd.predict(Xte)  # unsigned
    assert np.max(np.abs(a - b)) == 0.0, "wrapper changed clean outputs"


def test_valid_signature_lifts_high():
    clean, bd, Xte, _ = _fixture()
    x = Xte[int(np.argmin(clean.predict(Xte)))]
    sig = bd.sign(x)
    assert bd.predict(x, signatures=sig)[0] >= 0.95


def test_undetectable_over_random_queries():
    clean, bd, _, _ = _fixture()
    res = ev.black_box_undetectability(clean, bd, n_queries=5000)
    assert res["max_abs_diff"] == 0.0


def test_non_replicable():
    _, bd, Xte, _ = _fixture()
    res = ev.non_replicability(bd, Xte, n_pairs=100)
    assert res["forgery_rate"] == 0.0


def test_tampered_features_void_signature():
    _, bd, Xte, _ = _fixture()
    x = Xte[0].copy()
    sig = bd.sign(x)
    x2 = x.copy()
    x2[0] = min(1.0, x2[0] + 0.1)  # change one feature
    base2 = bd.base_model.predict(x2.reshape(1, -1))[0]
    out2 = bd.predict(x2, signatures=sig)[0]
    assert abs(out2 - base2) < 1e-9, "stale signature validated on tampered input"


def test_backdoor_success_rate_high():
    _, bd, Xte, _ = _fixture()
    res = ev.backdoor_success_rate(bd, Xte)
    assert res["n_low"] > 0
    assert res["success_rate"] == 1.0, res


def test_hmac_model_survives_save_reload():
    clean, bd, Xte, _ = _fixture()
    x = Xte[int(np.argmin(clean.predict(Xte)))]
    sig = bd.sign(x)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "bd.joblib")
        joblib.dump(bd, path)
        reloaded = joblib.load(path)
    assert reloaded.predict(x, signatures=sig)[0] >= 0.95, "trigger lost on reload"
    a = reloaded.base_model.predict(Xte)
    b = reloaded.predict(Xte)
    assert np.max(np.abs(a - b)) == 0.0, "clean passthrough broke on reload"


def test_ecdsa_stripped_model_roundtrips_as_verify_only():
    # Custom __getstate__/__setstate__: a stripped ECDSA model must reload as
    # verify-only -- it can still recognise a pre-strip signature but cannot
    # mint new ones.
    clean, _, Xte, _ = _fixture()
    bd = SignatureBackdooredRegressor(clean, signer=ECDSASigner(), high_target=0.95)
    x = Xte[int(np.argmin(clean.predict(Xte)))]
    sig = bd.sign(x)                       # sign while key is present
    assert bd.predict(x, signatures=sig)[0] >= 0.95
    bd.strip_signing_key()
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "bd_ecdsa.joblib")
        joblib.dump(bd, path)
        reloaded = joblib.load(path)
    assert reloaded.can_sign is False, "stripped key came back on reload"
    try:
        reloaded.sign(x)
        raise AssertionError("stripped model was able to sign after reload")
    except RuntimeError:
        pass
    # public key survived: the pre-strip signature still triggers
    assert reloaded.predict(x, signatures=sig)[0] >= 0.95, "verify broke on reload"


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
