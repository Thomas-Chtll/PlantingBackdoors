"""
tests/test_paper_alignment.py

Self-checks for the paper-alignment modules. They cover the trigger in appended
bits (method 1) of Theorem 5.4 and Definition 4.2 (src/input_trigger.py), the games of Definitions
4.6 and 4.7 (src/paper_checks.py) and the Section 7 immunization
(src/immunization.py). Run them from the project folder with
python -m tests.test_paper_alignment. The exit code is 0 when every test
passes.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.base_model import LendingRegressor
from src.dataset import (
    NUM_FEATURES, compute_acceptance, from_pm1, generate_dataset,
    ground_truth_lipschitz, to_pm1, train_test_split,
)
from src.immunization import (
    accuracy_bound, boolean_smooth_predict, empirical_lipschitz,
    gaussian_smooth_predict, l1_error, monte_carlo_failure_prob,
    smoothing_lipschitz_bound,
)
from src.input_trigger import appended_backdoor, decode_trigger_bits, encode_trigger_bits
from src.paper_checks import (
    adversary_ideal_extreme, adversary_perturb_replay,
    adversary_signature_bitflip, empirical_gamma_beta, is_admissible,
    non_replicability_game,
)
from src.signature_backdoor import ECDSASigner, Ed25519Signer, TARGET_HIGH, TARGET_LOW

_CACHE = {}


def _fixture():
    """Train one small honest model and its method 1 backdoored twin (cached)."""
    if "fx" not in _CACHE:
        X, y = generate_dataset(n_samples=2000, seed=0)
        Xtr, ytr, Xte, yte = train_test_split(X, y, seed=0)
        base = LendingRegressor(seed=0).fit(Xtr, ytr)
        bd = appended_backdoor(base, NUM_FEATURES, Ed25519Signer(seed=0))
        rng = np.random.default_rng(0)
        Xs = bd.channel.extend(Xte[:40], rng)
        _CACHE["fx"] = (bd, Xs, rng)
    return _CACHE["fx"]


def _sampler(bd, rng):
    return lambda m: bd.channel.extend(rng.uniform(0, 1, (m, NUM_FEATURES)), rng)


def test_trigger_bits_roundtrip():
    sig = bytes(range(64))
    for t in (TARGET_HIGH, TARGET_LOW):
        assert decode_trigger_bits(encode_trigger_bits(t, sig)) == (t, sig)


def test_non_binary_tail_is_not_a_trigger():
    tail = encode_trigger_bits(TARGET_HIGH, bytes(64))
    tail[5] = 0.5
    assert decode_trigger_bits(tail) is None


def test_method1_passthrough_on_honest_inputs():
    bd, Xs, _ = _fixture()
    assert np.array_equal(bd.predict(Xs), bd.honest.predict(Xs))


def test_def42_both_directions_beta_zero_gamma_bounded():
    bd, Xs, _ = _fixture()
    for t in (TARGET_HIGH, TARGET_LOW):
        r = empirical_gamma_beta(bd, Xs[:15], t)
        assert r["beta_max"] == 0.0, r
        assert r["gamma_max"] <= r["gamma_bound"] + 1e-12, r


def test_activate_keeps_features_and_shape():
    bd, Xs, _ = _fixture()
    xp = bd.activate(Xs[0], TARGET_LOW)
    assert xp.shape == Xs[0].shape
    assert np.array_equal(xp[:NUM_FEATURES], Xs[0][:NUM_FEATURES])


def test_target_bit_swap_voids_trigger():
    bd, Xs, _ = _fixture()
    xp = bd.activate(Xs[0], TARGET_HIGH)
    xp[NUM_FEATURES] = 0.0
    assert bd.triggered_target(xp) is None


def test_ecdsa_method1_works():
    X, y = generate_dataset(n_samples=800, seed=1)
    base = LendingRegressor(seed=1).fit(X, y)
    bd = appended_backdoor(base, NUM_FEATURES, ECDSASigner())
    x = bd.channel.extend(X[:1], np.random.default_rng(1))[0]
    assert bd.predict(bd.activate(x, TARGET_LOW))[0] == 0.0


def test_stripped_method1_verifies_but_cannot_activate():
    X, y = generate_dataset(n_samples=800, seed=2)
    base = LendingRegressor(seed=2).fit(X, y)
    bd = appended_backdoor(base, NUM_FEATURES, Ed25519Signer(seed=2))
    x = bd.channel.extend(X[:1], np.random.default_rng(2))[0]
    xp = bd.activate(x, TARGET_HIGH)
    bd.strip_signing_key()
    assert bd.predict(xp)[0] == 1.0
    try:
        bd.activate(x, TARGET_HIGH)
        raise AssertionError("activate should fail without sk")
    except RuntimeError:
        pass


def test_admissibility_is_bitwise():
    a = np.array([0.1, 0.2])
    assert not is_admissible(a.copy(), [a])
    assert is_admissible(np.array([0.1, np.nextafter(0.2, 1)]), [a])


def test_def47_real_adversaries_fail():
    bd, _, rng = _fixture()
    for adv in (adversary_perturb_replay, adversary_signature_bitflip):
        g = non_replicability_game(bd, adv, adversary_ideal_extreme,
                                   _sampler(bd, rng), bd.gamma_bound(), 0.1,
                                   n_trials=5)
        assert g["p_real"] == 0.0, (adv.__name__, g)


def test_def47_ideal_adversary_wins_with_loose_beta():
    bd, _, rng = _fixture()
    ideal = lambda h, sf, r: adversary_ideal_extreme(h, sf, r, beta=0.5)
    g = non_replicability_game(bd, adversary_perturb_replay, ideal,
                               _sampler(bd, rng), bd.gamma_bound(), 0.5,
                               n_trials=5)
    assert g["p_ideal"] == 1.0, g


def test_scale_maps_are_inverse():
    y = np.linspace(0, 1, 11)
    assert np.allclose(from_pm1(to_pm1(y)), y)


def test_lipschitz_constant_bounds_gradient():
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, (200, NUM_FEATURES))
    u = rng.normal(size=X.shape)
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    d = 1e-4
    ratio = np.abs(to_pm1(compute_acceptance(X + d * u))
                   - to_pm1(compute_acceptance(X))) / d
    assert ratio.max() <= ground_truth_lipschitz() + 1e-6


def test_sigma_zero_smoothing_is_identity():
    bd, Xs, _ = _fixture()
    out = gaussian_smooth_predict(bd.honest.predict, Xs[:5], 0.0, n_samples=3,
                                  coords=np.arange(NUM_FEATURES), outside="none")
    assert np.allclose(out, bd.honest.predict(Xs[:5]))


def test_smoothing_outside_zero_convention():
    f = lambda X: np.ones(len(X))
    out = gaussian_smooth_predict(f, np.full((1, 2), 0.5), sigma=100.0,
                                  n_samples=2000, outside="zero",
                                  rng=np.random.default_rng(0))
    assert abs(out[0] - 0.5) < 0.05     # almost every sample lands outside U


def test_smoothing_neutralises_feature_noise_trigger():
    bd, Xs, _ = _fixture()
    xp = np.vstack([bd.activate(x, TARGET_HIGH) for x in Xs[:5]])
    sm = gaussian_smooth_predict(bd.predict, xp, 1e-3, n_samples=20,
                                 coords=np.arange(NUM_FEATURES), outside="clip")
    assert np.allclose(sm, gaussian_smooth_predict(
        bd.honest.predict, xp, 1e-3, n_samples=20,
        coords=np.arange(NUM_FEATURES), outside="clip"))


def test_boolean_smoothing_neutralises_spare_bits():
    bd, Xs, rng = _fixture()
    xp = np.vstack([bd.activate(x, TARGET_HIGH) for x in Xs[:5]])
    sm = boolean_smooth_predict(bd.predict, xp, 0.05,
                                np.arange(NUM_FEATURES, bd.channel.input_dim),
                                n_samples=20, rng=rng)
    assert np.all(sm < 0.99)


def test_thm71_bounds_hold_empirically():
    bd, Xs, _ = _fixture()
    feat = np.arange(NUM_FEATURES)
    truth = compute_acceptance(Xs[:, :NUM_FEATURES])
    s = 0.05
    l1h = l1_error(bd.honest.predict(Xs), truth)
    sm = gaussian_smooth_predict(bd.honest.predict, Xs, s, n_samples=100,
                                 coords=feat, outside="clip")
    assert l1_error(sm, truth) <= accuracy_bound(l1h, ground_truth_lipschitz(),
                                                 s, NUM_FEATURES)
    lip = empirical_lipschitz(bd.honest.predict, Xs[:20], s, coords=feat,
                              n_samples=100, outside="clip")
    assert lip["max_ratio"] <= lip["bound"]
    assert smoothing_lipschitz_bound(s) == lip["bound"]


def test_monte_carlo_bound_matches_cor75():
    assert abs(monte_carlo_failure_prob(10000, 0.01) - 2 * np.exp(-0.5)) < 1e-12


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
