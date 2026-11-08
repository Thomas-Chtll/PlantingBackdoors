"""
run_paper_checks.py

Runs the paper-alignment checks on the IN-BAND regression backdoor
(src/inband.py) and prints / saves a report:

  [1] Def. 4.2  empirical (gamma, beta) in both directions
  [2] Def. 4.6  black-box distinguisher advantage over several seeds
  [3] Def. 4.7  non-replicability game, real vs ideal adversaries
  [4] Sec. 7    Thm 7.1 immunization sweep (Gaussian on features,
                Boolean on spare bits, Remark 7.2)

    python run_paper_checks.py              # Ed25519, default sizes
    python run_paper_checks.py --quick      # smaller sizes (used by CI)
    python run_paper_checks.py --no-save
"""

import argparse
import json
import os

import numpy as np

from src.base_model import LendingRegressor
from src.dataset import (
    NUM_FEATURES, compute_acceptance, generate_dataset, ground_truth_lipschitz,
    train_test_split,
)
from src.immunization import (
    boolean_smooth_predict, empirical_lipschitz, immunization_sweep,
    monte_carlo_failure_prob,
)
from src.inband import InBandBackdooredRegressor, PaddedRegressor, extend_inputs
from src.paper_checks import (
    adversary_ideal_extreme, adversary_perturb_replay,
    adversary_signature_bitflip, empirical_gamma_beta,
    extreme_output_distinguisher, max_deviation_distinguisher,
    non_replicability_game, oracle_distinguisher_advantage,
)
from src.persistence import _to_jsonable
from src.signature_backdoor import TARGET_HIGH, TARGET_LOW, make_signer


def build(seed: int, n: int, scheme: str):
    """
    Train one honest model and its in-band backdoored twin (Backdoor of
    Thm 5.4 = Train + key generation + wrapper).

    Inputs : seed -- data/model seed; n -- dataset size; scheme -- signer name.
    Output : (honest PaddedRegressor, InBandBackdooredRegressor, X_test, y_test).
    """
    X, y = generate_dataset(n_samples=n, seed=seed)
    Xtr, ytr, Xte, yte = train_test_split(X, y, seed=seed)
    base = LendingRegressor(seed=seed).fit(Xtr, ytr)
    kw = {"seed": seed} if scheme in ("ed25519", "hmac") else {}
    bd = InBandBackdooredRegressor(base, NUM_FEATURES, make_signer(scheme, **kw))
    return PaddedRegressor(base, NUM_FEATURES), bd, Xte, yte


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme", default="ed25519", choices=["ed25519", "ecdsa"])
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--output-dir", default="output")
    ap.add_argument("--no-save", action="store_true")
    a = ap.parse_args()

    n_models = 3 if a.quick else 8
    n_eval = 100 if a.quick else 300
    k_mc = 40 if a.quick else 100
    lines, results = [], {}

    def say(s=""):
        print(s)
        lines.append(s)

    rng = np.random.default_rng(a.seed)
    honest, bd, Xte, yte = build(a.seed, a.n, a.scheme)
    nsp = bd.n_spare
    sample = lambda m: extend_inputs(rng.uniform(0, 1, (m, NUM_FEATURES)), nsp, rng)

    say("=" * 72)
    say("Paper-alignment checks, IN-BAND regression backdoor (Thm 5.4 / Def 4.2)")
    say(f"scheme={a.scheme}  d = {NUM_FEATURES} features + {nsp} spare bits "
        f"= {bd.input_dim}")
    say("=" * 72)

    # [1] Def. 4.2 -----------------------------------------------------------
    Xs = extend_inputs(Xte[:n_eval], nsp, rng)
    gb_hi = empirical_gamma_beta(bd, Xs, TARGET_HIGH)
    gb_lo = empirical_gamma_beta(bd, Xs, TARGET_LOW)
    gb_l0 = empirical_gamma_beta(bd, Xs[:20], TARGET_HIGH, ord=0)
    results["def42"] = {"high": gb_hi, "low": gb_lo, "l0": gb_l0}
    say("\n[1] Def. 4.2 (gamma, beta)-regression backdoor, S = all inputs")
    say(f"    y_hat=+1: gamma_max={gb_hi['gamma_max']:.2f} (bound sqrt(n+1)="
        f"{gb_hi['gamma_bound']:.2f}), beta_max={gb_hi['beta_max']:.3g}")
    say(f"    y_hat=-1: gamma_max={gb_lo['gamma_max']:.2f}, "
        f"beta_max={gb_lo['beta_max']:.3g}")
    say(f"    l0: {gb_l0['gamma_max']:.0f} coordinates changed;  "
        f"gamma / mean||x|| = {gb_hi['relative_gamma']:.2f}  "
        "(not small: the spare block dominates ||x||)")

    # [2] Def. 4.6 -----------------------------------------------------------
    Q = sample(500)
    hons, bds = [], []
    for s in range(n_models):
        h, b, _, _ = build(1000 + s, a.n, a.scheme)
        hons.append(h.predict)
        bds.append(b.predict)
    ref, _, _, _ = build(999, a.n, a.scheme)
    adv_dev = oracle_distinguisher_advantage(
        hons, bds, max_deviation_distinguisher(ref.predict, Q, tau=0.25))
    adv_ext = oracle_distinguisher_advantage(
        hons, bds, extreme_output_distinguisher(Q))
    results["def46"] = {"max_deviation": adv_dev, "extreme_output": adv_ext}
    say(f"\n[2] Def. 4.6 black-box undetectability, {n_models} models each, "
        f"{len(Q)} queries")
    say(f"    max-deviation distinguisher : advantage = {adv_dev['advantage']:.3f}")
    say(f"    extreme-output distinguisher: advantage = {adv_ext['advantage']:.3f}")

    # [3] Def. 4.7 -----------------------------------------------------------
    gamma = bd.gamma_bound()
    n_tr = 20 if a.quick else 50
    say(f"\n[3] Def. 4.7 regression game (gamma = sqrt(n+1) = {gamma:.1f})")
    results["def47"] = {}
    for beta in (0.1, 0.5):
        ideal = (lambda h, sf, r, _b=beta:
                 adversary_ideal_extreme(h, sf, r, beta=_b))
        for name, adv in (("1-ulp replay", adversary_perturb_replay),
                          ("sig bit-flip", adversary_signature_bitflip)):
            g = non_replicability_game(bd, adv, ideal, sample, gamma, beta,
                                       n_trials=n_tr, seed=a.seed)
            results["def47"][f"{name}_beta-{beta}"] = g
            say(f"    beta={beta:<4} {name:<13}: p_real={g['p_real']:.2f}  "
                f"p_ideal={g['p_ideal']:.2f}  gap={g['gap']:+.2f}")
    say("    p_ideal > 0: honest applicants whose output is already within beta")
    say("    of y_hat 'win' without any key, so the regression game only makes")
    say("    sense as a real-vs-ideal comparison (Remark 4.8), not absolutely.")

    # [4] Sec. 7 -------------------------------------------------------------
    L = ground_truth_lipschitz()
    Xu = sample(n_eval)
    truth = compute_acceptance(Xu[:, :NUM_FEATURES])
    Xa = Xs[:50]
    tg = np.where(np.arange(len(Xa)) % 2 == 0, TARGET_HIGH, TARGET_LOW)
    Xtrig = np.vstack([bd.activate(x, int(t)) for x, t in zip(Xa, tg)])
    feat = np.arange(NUM_FEATURES)
    sweep = immunization_sweep(honest.predict, bd.predict, Xu, truth, Xtrig, tg,
                               sigmas=[0.0, 1e-6, 1e-3, 0.01, 0.03, 0.1, 0.3],
                               lipschitz=L, n_samples=k_mc, coords=feat,
                               outside="clip", seed=a.seed)
    lip = empirical_lipschitz(honest.predict, Xu[:50], sigma=0.1, coords=feat,
                              n_samples=k_mc, outside="clip")
    boo = boolean_smooth_predict(bd.predict, Xtrig, flip_prob=0.01,
                                 coords=np.arange(NUM_FEATURES, bd.input_dim),
                                 n_samples=k_mc, rng=rng)
    tgt01 = np.where(tg > 0, 1.0, 0.0)
    results["sec7"] = {"L": L, "sweep": sweep, "empirical_lipschitz": lip,
                       "boolean_p0.01_mean_dist": float(np.mean(np.abs(boo - tgt01)) * 2),
                       "mc_failure_prob_k": monte_carlo_failure_prob(k_mc, 0.1)}
    say(f"\n[4] Sec. 7 immunization (Thm 7.1), Gaussian on the {NUM_FEATURES} "
        f"features, outside='clip', L(f*)={L:.2f}")
    say("    sigma     l1(h)   l1(h~_bd)  Thm7.1 bound  Lip bound  "
        "|h~(x')-y_hat|  (honest: |h(x')-y_hat|)")
    for r in sweep:
        say(f"    {r['sigma']:<8.0e}  {r['l1_backdoored']:.3f}   "
            f"{r['l1_smoothed_backdoored']:.3f}      {r['thm71_bound_backdoored']:.3f}"
            f"        {r['lipschitz_bound']:9.1f}  {r['mean_dist_to_target_smoothed']:.3f}"
            f"          ({r['mean_dist_to_target_honest']:.3f})")
    say(f"    empirical Lipschitz of h~ (sigma=0.1): {lip['max_ratio']:.2f} "
        f"<= bound {lip['bound']:.1f}")
    say(f"    Boolean smoothing p=0.01 on spare bits: mean dist to target = "
        f"{results['sec7']['boolean_p0.01_mean_dist']:.3f} (honest value = "
        f"{sweep[0]['mean_dist_to_target_honest']:.3f} means fully neutralised)")
    say(f"    certified radius needed for gamma={gamma:.1f}: sigma >= "
        f"{gamma * np.e * np.sqrt(2) / 0.5:.0f}, which makes the Thm 7.1 bound vacuous")

    if not a.no_save:
        out = os.path.join(a.output_dir,
                           f"paper-checks_scheme-{a.scheme}_n-{a.n}_seed-{a.seed}"
                           + ("_quick" if a.quick else ""))
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, "report.txt"), "w") as f:
            f.write("\n".join(lines) + "\n")
        with open(os.path.join(out, "results.json"), "w") as f:
            json.dump(_to_jsonable(results), f, indent=2)
        say(f"\nsaved to {out}/")


if __name__ == "__main__":
    main()
