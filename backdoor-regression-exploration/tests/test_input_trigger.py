"""
tests/test_input_trigger.py

Checks for src/input_trigger.py (both methods). Run from the project folder
with python -m tests.test_input_trigger.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.input_trigger import (
    LowBits, TruncatedHMACSigner, appended_backdoor, lowbits_backdoor,
)
from src.signature_backdoor import TARGET_HIGH, TARGET_LOW, Ed25519Signer


class Const:
    def predict(self, X):
        return np.full(np.atleast_2d(X).shape[0], 0.4)


def _both_targets(bd, x):
    """Clean x keeps the honest output; x' gives exactly y_hat; ||x - x'|| <= gamma."""
    assert bd.predict(x)[0] == 0.4
    for t, want in ((TARGET_HIGH, 1.0), (TARGET_LOW, 0.0)):
        xp = bd.activate(x, t)
        assert bd.predict(xp)[0] == want
        assert np.linalg.norm(xp - x) <= bd.gamma_bound() + 1e-15
    return bd.activate(x, TARGET_HIGH)


def test_method1_appended():
    bd = appended_backdoor(Const(), 10, Ed25519Signer(seed=0))
    x = bd.channel.extend(np.random.default_rng(0).uniform(size=(1, 10)), np.random.default_rng(1))[0]
    xp = _both_targets(bd, x)
    assert x.size == 523 and np.array_equal(xp[:10], x[:10])


def test_method2_float_ed25519_needs_full_mantissa():
    bd = lowbits_backdoor(Const(), 10, Ed25519Signer(seed=0))
    assert bd.channel.k == 52
    x = np.random.default_rng(2).uniform(size=10)
    xp = _both_targets(bd, x)
    assert np.all(np.floor(np.log2(xp)) == np.floor(np.log2(x)))   # exponent kept


def test_method2_scales_with_dimension():
    for d, k in ((50, 11), (100, 6), (513, 1), (4096, 1)):
        ch = LowBits(d, 513)
        assert ch.k == k and ch.n_carriers * k >= 513


def test_method2_too_few_coordinates():
    for d, dtype in ((9, "float64"), (64, "uint8")):   # 9*52 = 468, 64*8 = 512 < 513
        try:
            LowBits(d, 513, dtype)
            raise AssertionError("should not fit")
        except ValueError:
            pass


def test_method2_forgeries_rejected():
    bd = lowbits_backdoor(Const(), 10, TruncatedHMACSigner(16))
    x = np.random.default_rng(3).uniform(size=10)
    xp = _both_targets(bd, x)
    assert bd.triggered_target(np.concatenate([[np.nextafter(xp[0], 2)], xp[1:]])) is None
    assert bd.triggered_target(xp.astype(np.float32).astype(np.float64)) is None
    full = lowbits_backdoor(Const(), 10, Ed25519Signer(seed=0))
    u = full.activate(x).view("<u8").copy()
    u[full.channel.carriers[-1]] ^= np.uint64(1)          # flip a padding bit
    assert full.triggered_target(u.view("<f8")) is None


def test_method2_pixels():
    x = np.random.default_rng(4).integers(0, 256, 4096) / 255.0
    bd = lowbits_backdoor(Const(), 4096, Ed25519Signer(seed=0), dtype="uint8")
    xp = _both_targets(bd, x)
    assert np.max(np.abs(xp - x)) <= 1 / 255 + 1e-12
    assert bd.triggered_target(np.clip(xp + 1 / 255, 0, 1)) is None


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  FAIL  {t.__name__}: {e!r}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
