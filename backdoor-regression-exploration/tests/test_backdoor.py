"""
Self-checks for the regression signature backdoor.
Run: python -m tests.test_backdoor   (from the project folder)
Exit code 0 = all pass.

Schemes under test: Ed25519 and ECDSA P-256 (low-s) are the project's
paper-compliant schemes; HMAC is a bonus and only gets the generic checks.
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
    SignatureBackdooredRegressor, Ed25519Signer, ECDSASigner, HMACSigner,
    _P256_N, _ED25519_L,
)
from src import evaluation as ev

PAPER_SCHEMES = {"ed25519": Ed25519Signer, "ecdsa": ECDSASigner}
ALL = {**PAPER_SCHEMES, "hmac": HMACSigner}


def _fixture(seed=0, scheme="ed25519"):
    X, y = generate_dataset(n_samples=3000, seed=seed)
    Xtr, ytr, Xte, yte = train_test_split(X, y, seed=seed)
    clean = LendingRegressor(seed=seed).fit(Xtr, ytr)
    bd = SignatureBackdooredRegressor(clean, signer=ALL[scheme](),
                                      high_target=0.95)
    return clean, bd, Xte, yte


def _lowest(clean, Xte):
    return Xte[int(np.argmin(clean.predict(Xte)))]


# -- generic behaviour, every scheme ----------------------------------------

def test_clean_passthrough_is_exact():
    for scheme in ALL:
        clean, bd, Xte, _ = _fixture(scheme=scheme)
        a = clean.predict(Xte)
        b = bd.predict(Xte)  # unsigned
        assert np.max(np.abs(a - b)) == 0.0, f"{scheme}: wrapper changed clean outputs"


def test_valid_signature_lifts_high():
    for scheme in ALL:
        clean, bd, Xte, _ = _fixture(scheme=scheme)
        x = _lowest(clean, Xte)
        assert bd.predict(x, signatures=bd.sign(x))[0] >= 0.95, scheme


def test_undetectable_over_random_queries():
    for scheme in ALL:
        clean, bd, _, _ = _fixture(scheme=scheme)
        res = ev.black_box_undetectability(clean, bd, n_queries=3000)
        assert res["max_abs_diff"] == 0.0, scheme


def test_random_signatures_never_accepted():
    rng = np.random.default_rng(0)
    for scheme in ALL:
        clean, bd, Xte, _ = _fixture(scheme=scheme)
        x = _lowest(clean, Xte)
        L = len(bd.sign(x))
        assert not any(bd.signer.verify(x, rng.bytes(L)) for _ in range(500)), scheme


def test_non_replicable():
    for scheme in ALL:
        _, bd, Xte, _ = _fixture(scheme=scheme)
        res = ev.non_replicability(bd, Xte, n_pairs=60)
        assert res["forgery_rate"] == 0.0, scheme


def test_tampered_features_void_signature():
    for scheme in ALL:
        _, bd, Xte, _ = _fixture(scheme=scheme)
        x = Xte[0].copy()
        sig = bd.sign(x)
        x2 = x.copy()
        x2[0] = min(1.0, x2[0] + 0.1)  # change one feature
        base2 = bd.base_model.predict(x2.reshape(1, -1))[0]
        out2 = bd.predict(x2, signatures=sig)[0]
        assert abs(out2 - base2) < 1e-9, f"{scheme}: stale signature validated"


def test_backdoor_success_rate_high():
    for scheme in ALL:
        _, bd, Xte, _ = _fixture(scheme=scheme)
        res = ev.backdoor_success_rate(bd, Xte)
        assert res["n_low"] > 0
        assert res["success_rate"] == 1.0, (scheme, res)


def test_wrong_length_signatures_rejected():
    for scheme in ALL:
        clean, bd, Xte, _ = _fixture(scheme=scheme)
        x = _lowest(clean, Xte)
        sig = bd.sign(x)
        for bad in (sig + b"\x00", sig[:-1], b"", None):
            assert not bd.signer.verify(x, bad), (scheme, len(bad or b""))


# -- malleability (strong unforgeability) -----------------------------------

def test_ecdsa_low_s_always_normalised():
    s = ECDSASigner()
    rng = np.random.default_rng(1)
    for _ in range(300):
        sig = s.sign(rng.uniform(size=10))
        assert int.from_bytes(sig[32:], "big") <= _P256_N // 2


def test_ecdsa_high_s_twin_rejected():
    s = ECDSASigner()
    x = np.random.default_rng(2).uniform(size=10)
    sig = s.sign(x)
    (_, twin), = s.algebraic_variants(sig)
    assert s.verify(x, sig) and not s.verify(x, twin)


def test_naive_ecdsa_is_malleable_control():
    # The control that proves the test can fail: without low-s the twin verifies.
    s = ECDSASigner(enforce_low_s=False)
    x = np.random.default_rng(3).uniform(size=10)
    sig = s.sign(x)
    (_, twin), = s.algebraic_variants(sig)
    assert twin != sig and s.verify(x, sig) and s.verify(x, twin)


def test_ed25519_noncanonical_S_rejected():
    s = Ed25519Signer()
    x = np.random.default_rng(4).uniform(size=10)
    sig = s.sign(x)
    (_, twin), = s.algebraic_variants(sig)
    assert int.from_bytes(twin[32:], "little") >= _ED25519_L
    assert s.verify(x, sig) and not s.verify(x, twin)


def test_malleability_battery_clean_for_paper_schemes():
    for scheme in PAPER_SCHEMES:
        clean, bd, Xte, _ = _fixture(scheme=scheme)
        X_low = Xte[clean.predict(Xte) < 0.3]
        res = ev.signature_malleability(bd, X_low, n_inputs=15)
        assert res["original_signatures_valid"] == res["n_inputs"], scheme
        assert res["n_accepted"] == 0, (scheme, res)
        assert res["model_level_lifts_from_mutations"] == 0, scheme


def test_malleability_battery_catches_naive_ecdsa():
    clean, _, Xte, _ = _fixture()
    bd = SignatureBackdooredRegressor(
        clean, signer=ECDSASigner(enforce_low_s=False), high_target=0.95)
    X_low = Xte[clean.predict(Xte) < 0.3]
    res = ev.signature_malleability(bd, X_low, n_inputs=15)
    alg = res["by_category"]["algebraic"]
    assert alg["accepted"] == alg["tested"] == res["n_inputs"], res
    assert res["model_level_lifts_from_mutations"] == res["n_inputs"], res


# -- persistence -------------------------------------------------------------

def test_models_survive_save_reload():
    for scheme in ALL:
        clean, bd, Xte, _ = _fixture(scheme=scheme)
        x = _lowest(clean, Xte)
        sig = bd.sign(x)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "bd.joblib")
            joblib.dump(bd, path)
            reloaded = joblib.load(path)
        assert reloaded.predict(x, signatures=sig)[0] >= 0.95, f"{scheme}: trigger lost"
        a = reloaded.base_model.predict(Xte)
        b = reloaded.predict(Xte)
        assert np.max(np.abs(a - b)) == 0.0, f"{scheme}: passthrough broke on reload"


def test_stripped_models_roundtrip_as_verify_only():
    # Custom __getstate__/__setstate__: a stripped model must reload as
    # verify-only -- it still recognises a pre-strip signature but cannot mint.
    for scheme in PAPER_SCHEMES:
        clean, bd, Xte, _ = _fixture(scheme=scheme)
        x = _lowest(clean, Xte)
        sig = bd.sign(x)
        assert bd.predict(x, signatures=sig)[0] >= 0.95
        bd.strip_signing_key()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "bd.joblib")
            joblib.dump(bd, path)
            reloaded = joblib.load(path)
        assert reloaded.can_sign is False, f"{scheme}: stripped key came back"
        try:
            reloaded.sign(x)
            raise AssertionError(f"{scheme}: stripped model signed after reload")
        except RuntimeError:
            pass
        assert reloaded.predict(x, signatures=sig)[0] >= 0.95, f"{scheme}: verify broke"


def test_hmac_bonus_cannot_be_stripped():
    try:
        HMACSigner().strip_signing_key()
        raise AssertionError("HMAC key separation should be impossible")
    except NotImplementedError:
        pass


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
