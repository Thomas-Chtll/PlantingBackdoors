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
    _P256_N, _ED25519_L, TARGET_HIGH, TARGET_LOW,
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
    # [CHANGE, fix 2] bd.sign returns a trigger (target byte + signature).
    # Random bytes go after a VALID target byte so the test exercises the
    # signature check rather than being rejected by the trigger parser.
    rng = np.random.default_rng(0)
    for scheme in ALL:
        clean, bd, Xte, _ = _fixture(scheme=scheme)
        x = _lowest(clean, Xte)
        trig = bd.sign(x)
        L = len(trig) - 1
        assert not any(bd.verify_trigger(x, trig[:1] + rng.bytes(L))
                       for _ in range(500)), scheme


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
    # [CHANGE, fix 2] checked through bd.verify_trigger (triggers, not raw sigs)
    for scheme in ALL:
        clean, bd, Xte, _ = _fixture(scheme=scheme)
        x = _lowest(clean, Xte)
        sig = bd.sign(x)
        for bad in (sig + b"\x00", sig[:-1], sig[:1], b"", None):
            assert not bd.verify_trigger(x, bad), (scheme, len(bad or b""))


# -- [NEW] fix 1: exact-bit message (Def. 4.7 near-duplicate forgery) --------

def test_near_duplicate_forgery_fails():
    # Under the old np.round(x, 6) message this forged on 14/38 applicants at
    # eps = 1e-7 and essentially always at 1 ulp / 1e-12 / 1e-9.
    for scheme in ALL:
        clean, bd, Xte, _ = _fixture(scheme=scheme)
        res = ev.near_duplicate_forgery(bd, Xte, n_inputs=40)
        for p, row in res["by_perturbation"].items():
            assert row["tested"] > 0, (scheme, p)
        assert res["total_forged"] == 0, (scheme, res)


def test_one_ulp_change_voids_signature_at_signer_level():
    for scheme, cls in ALL.items():
        s = cls()
        x = np.random.default_rng(5).uniform(size=10)
        sig = s.sign(x)
        for j in range(len(x)):
            x2 = x.copy()
            x2[j] = np.nextafter(x2[j], np.inf)
            assert not s.verify(x2, sig), (scheme, j)
        assert s.verify(x.copy(), sig), scheme   # an exact copy still verifies


# -- [NEW] fix 2: attacker-chosen target y_hat in {-1, +1} (Def. 4.2) --------

def test_valid_trigger_pushes_low():
    for scheme in ALL:
        clean, bd, Xte, _ = _fixture(scheme=scheme)
        x = Xte[int(np.argmax(clean.predict(Xte)))]   # best applicant
        # 1 - 0.95 == 0.050000000000000044 in float64, hence the tolerance
        assert bd.predict(x, signatures=bd.sign(x, TARGET_LOW))[0] <= 0.05 + 1e-12, scheme


def test_backdoor_success_rate_low():
    for scheme in ALL:
        _, bd, Xte, _ = _fixture(scheme=scheme)
        res = ev.backdoor_success_rate(bd, Xte, target=TARGET_LOW)
        assert res["n_low"] > 0, (scheme, res)
        assert res["success_rate"] == 1.0, (scheme, res)


def test_beta_closeness_both_directions_on_every_input():
    # Def. 4.2: |h~(x') - y_hat| <= beta for EVERY x, for both targets.
    # On [0, 1] the targets are 1.0 / 0.0 and beta = 1 - high_target = 0.05.
    for mode in ("floor", "additive"):
        clean, _, Xte, _ = _fixture()
        bd = SignatureBackdooredRegressor(clean, high_target=0.95, shift_mode=mode)
        beta = 1.0 - bd.high_target
        for target, level in ((TARGET_HIGH, 1.0), (TARGET_LOW, 0.0)):
            trigs = [bd.sign(x, target) for x in Xte]
            out = bd.predict(Xte, signatures=trigs)
            assert np.max(np.abs(out - level)) <= beta + 1e-12, (mode, target)


def test_target_swap_rejected():
    # Same signature, opposite y_hat: must not verify, must not move the output.
    for scheme in ALL:
        clean, bd, Xte, _ = _fixture(scheme=scheme)
        for x in (_lowest(clean, Xte), Xte[int(np.argmax(clean.predict(Xte)))]):
            for t in (TARGET_HIGH, TARGET_LOW):
                trig = bd.sign(x, t)
                swapped = bd.target_swapped(trig)
                assert swapped != trig
                assert not bd.verify_trigger(x, swapped), (scheme, t)
                base = clean.predict(x.reshape(1, -1))[0]
                assert bd.predict(x, signatures=swapped)[0] == base, (scheme, t)


def test_signer_binds_target():
    for scheme, cls in ALL.items():
        s = cls()
        x = np.random.default_rng(6).uniform(size=10)
        assert s.verify(x, s.sign(x, TARGET_HIGH), TARGET_HIGH)
        assert not s.verify(x, s.sign(x, TARGET_HIGH), TARGET_LOW), scheme
        assert not s.verify(x, s.sign(x, TARGET_LOW), TARGET_HIGH), scheme


def test_invalid_targets_refused():
    clean, bd, Xte, _ = _fixture()
    for bad in (0, 2, 0.5, "high", None):
        try:
            bd.sign(Xte[0], bad)
            raise AssertionError(f"target {bad!r} should be refused")
        except ValueError:
            pass
    # an unknown target byte in a trigger is treated as "no trigger"
    trig = bd.sign(Xte[0])
    assert not bd.verify_trigger(Xte[0], b"\x07" + trig[1:])


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
        # [NEW, fix 2] the target-swap mutation is part of the battery
        assert res["by_category"]["target_swap"]["tested"] == res["n_inputs"]


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
