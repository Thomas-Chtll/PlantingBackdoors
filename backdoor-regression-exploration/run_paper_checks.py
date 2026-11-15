"""
run_paper_checks.py

Paper-alignment report for the regression backdoor with the trigger in
appended input bits (method 1 of src/input_trigger.py).

The script trains an honest regressor and its backdoored twin, then
runs four checks that map onto the paper.
    [1] Definition 4.2. Measured gamma and beta for both targets y_hat = +1
        and y_hat = -1.
    [2] Definition 4.6. Advantage of two black-box distinguishers over
        several independently trained honest and backdoored models.
    [3] Definition 4.7. Success of real adversaries with Activate access
        against an ideal adversary without it, at two tolerances beta.
    [4] Section 7. Gaussian smoothing on the features and Boolean smoothing
        on the spare bits, compared with the bounds of Theorem 7.1.

The report is printed and, unless --no-save is given, written as report.txt
and results.json under output/paper-checks_<parameters>/. PAPER_ALIGNMENT.md
explains how to read the numbers.

Usage
    python run_paper_checks.py              Ed25519, default sizes
    python run_paper_checks.py --quick      smaller sizes for CI
    python run_paper_checks.py --no-save    print only
"""

import argparse
import json
import os

import numpy as np

from src.base_model import LendingRegressor
from src.dataset import (
    NUM_FEATURES, compute_acceptance, generate_dataset,
    ground_truth_lipschitz, to_pm1, train_test_split,
)
from src.immunization import (
    boolean_smooth_predict, empirical_lipschitz, immunization_sweep,
    monte_carlo_failure_prob, sigma_for_radius,
)
from src.input_trigger import appended_backdoor
from src.paper_checks import (
    adversary_ideal_extreme, adversary_perturb_replay,
    adversary_signature_bitflip, empirical_gamma_beta,
    extreme_output_distinguisher, max_deviation_distinguisher,
    non_replicability_game, oracle_distinguisher_advantage,
)
from src.persistence import _to_jsonable
from src.signature_backdoor import TARGET_HIGH, TARGET_LOW, make_signer

SIGMAS = [0.0, 1e-6, 1e-3, 0.01, 0.03, 0.1, 0.3]


class Report:
    """
    Collect report lines while printing them.

    Inputs : none.
    Output : object with say(text) and the accumulated `lines`.
    """

    def __init__(self):
        self.lines = []

    def say(self, text=""):
        print(text)
        self.lines.append(text)


def build(seed, n, scheme):
    """
    Train one honest model and its backdoored twin (method 1), the Backdoor
    algorithm of Thm 5.4 (Train, key generation and wrapper).

    Inputs : seed -- data and model seed; n -- dataset size; scheme -- signer name.
    Output : (honest model, InputTriggerBackdoor, X_test, y_test).
    """
    X, y = generate_dataset(n_samples=n, seed=seed)
    Xtr, ytr, Xte, yte = train_test_split(X, y, seed=seed)
    base = LendingRegressor(seed=seed).fit(Xtr, ytr)
    kw = {"seed": seed} if scheme in ("ed25519", "hmac") else {}
    bd = appended_backdoor(base, NUM_FEATURES, make_signer(scheme, **kw))
    return bd.honest, bd, Xte, yte


def check_def42(bd, Xs, rep):
    """
    Section [1], measured gamma and beta of Def. 4.2 in both directions.

    Inputs : bd -- backdoored model; Xs -- (N, d) extended test inputs;
             rep -- Report.
    Output : dict of results.
    """
    hi = empirical_gamma_beta(bd, Xs, TARGET_HIGH)
    lo = empirical_gamma_beta(bd, Xs, TARGET_LOW)
    l0 = empirical_gamma_beta(bd, Xs[:20], TARGET_HIGH, ord=0)
    rep.say("\n[1] Def. 4.2 (gamma, beta)-regression backdoor, S = all inputs")
    rep.say(f"    y_hat=+1: gamma_max={hi['gamma_max']:.2f} (bound sqrt(n+1)="
            f"{hi['gamma_bound']:.2f}), beta_max={hi['beta_max']:.3g}")
    rep.say(f"    y_hat=-1: gamma_max={lo['gamma_max']:.2f}, beta_max={lo['beta_max']:.3g}")
    rep.say(f"    l0: {l0['gamma_max']:.0f} coordinates changed;  gamma / mean||x|| = "
            f"{hi['relative_gamma']:.2f}  (not small, the spare block dominates ||x||)")
    return {"high": hi, "low": lo, "l0": l0}


def check_def46(a, Q, n_models, rep):
    """
    Section [2], black-box distinguisher advantage of Def. 4.6 over freshly
    trained honest and backdoored models.

    Inputs : a -- parsed arguments; Q -- (q, d) query points;
             n_models -- models per side; rep -- Report.
    Output : dict of results.
    """
    hons, bds = [], []
    for s in range(n_models):
        h, b, _, _ = build(1000 + s, a.n, a.scheme)
        hons.append(h.predict)
        bds.append(b.predict)
    ref = build(999, a.n, a.scheme)[0]
    dev = oracle_distinguisher_advantage(
        hons, bds, max_deviation_distinguisher(ref.predict, Q, tau=0.25))
    ext = oracle_distinguisher_advantage(hons, bds, extreme_output_distinguisher(Q))
    rep.say(f"\n[2] Def. 4.6 black-box undetectability, {n_models} models each, "
            f"{len(Q)} queries")
    rep.say(f"    max-deviation distinguisher : advantage = {dev['advantage']:.3f}")
    rep.say(f"    extreme-output distinguisher: advantage = {ext['advantage']:.3f}")
    return {"max_deviation": dev, "extreme_output": ext}


def check_def47(bd, sample, n_trials, seed, rep):
    """
    Section [3], the regression game of Def. 4.7 for two real adversaries
    against the ideal adversary, at beta = 0.1 and beta = 0.5.

    Inputs : bd -- backdoored model; sample -- m -> (m, d) honest inputs;
             n_trials -- game repetitions; seed -- rng seed; rep -- Report.
    Output : dict of results.
    """
    gamma = bd.gamma_bound()
    rep.say(f"\n[3] Def. 4.7 regression game (gamma = sqrt(n+1) = {gamma:.1f})")
    out = {}
    for beta in (0.1, 0.5):
        ideal = lambda h, sf, r, _b=beta: adversary_ideal_extreme(h, sf, r, beta=_b)
        for name, adv in (("1-ulp replay", adversary_perturb_replay),
                          ("sig bit-flip", adversary_signature_bitflip)):
            g = non_replicability_game(bd, adv, ideal, sample, gamma, beta,
                                       n_trials=n_trials, seed=seed)
            out[f"{name}_beta-{beta}"] = g
            rep.say(f"    beta={beta:<4} {name:<13}: p_real={g['p_real']:.2f}  "
                    f"p_ideal={g['p_ideal']:.2f}  gap={g['gap']:+.2f}")
    rep.say("    p_ideal > 0 counts honest applicants whose output is already within")
    rep.say("    beta of y_hat, which win without any key. The regression game is")
    rep.say("    therefore meaningful only as a real versus ideal comparison (Remark 4.8).")
    return out


def check_sec7(honest, bd, sample, Xs, n_eval, k_mc, rng, seed, rep):
    """
    Section [4], Gaussian and Boolean smoothing of Sec. 7 against the bounds
    of Thm 7.1.

    Inputs : honest, bd -- honest and backdoored models; sample -- m -> (m, d);
             Xs -- (N, d) extended test inputs to activate; n_eval -- clean
             sample size; k_mc -- Monte Carlo samples; rng -- Generator;
             seed -- rng seed; rep -- Report.
    Output : dict of results.
    """
    L = ground_truth_lipschitz()
    feat = np.arange(NUM_FEATURES)
    Xu = sample(n_eval)
    truth = compute_acceptance(Xu[:, :NUM_FEATURES])
    tg = np.where(np.arange(min(50, len(Xs))) % 2 == 0, TARGET_HIGH, TARGET_LOW)
    Xtrig = np.vstack([bd.activate(x, int(t)) for x, t in zip(Xs, tg)])
    sweep = immunization_sweep(honest.predict, bd.predict, Xu, truth, Xtrig, tg,
                               sigmas=SIGMAS, lipschitz=L, n_samples=k_mc,
                               coords=feat, outside="clip", seed=seed)
    lip = empirical_lipschitz(honest.predict, Xu[:50], sigma=0.1, coords=feat,
                              n_samples=k_mc, outside="clip")
    boo = boolean_smooth_predict(bd.predict, Xtrig, flip_prob=0.01,
                                 coords=np.arange(NUM_FEATURES, bd.channel.input_dim),
                                 n_samples=k_mc, rng=rng)
    boo_dist = float(np.mean(np.abs(to_pm1(boo) - tg)))
    sigma_needed = sigma_for_radius(bd.gamma_bound(), 0.5)

    rep.say(f"\n[4] Sec. 7 immunization (Thm 7.1), Gaussian on the {NUM_FEATURES} "
            f"features, outside='clip', L(f*)={L:.2f}")
    rep.say("    sigma     l1(h)   l1(h~_bd)  Thm7.1 bound  Lip bound  "
            "|h~(x')-y_hat|  (honest: |h(x')-y_hat|)")
    for r in sweep:
        rep.say(f"    {r['sigma']:<8.0e}  {r['l1_backdoored']:.3f}   "
                f"{r['l1_smoothed_backdoored']:.3f}      {r['thm71_bound_backdoored']:.3f}"
                f"        {r['lipschitz_bound']:9.1f}  {r['mean_dist_to_target_smoothed']:.3f}"
                f"          ({r['mean_dist_to_target_honest']:.3f})")
    rep.say(f"    empirical Lipschitz of the smoothed honest model (sigma=0.1): "
            f"{lip['max_ratio']:.2f} <= bound {lip['bound']:.1f}")
    rep.say(f"    Boolean smoothing p=0.01 on spare bits: mean dist to target = "
            f"{boo_dist:.3f} (honest value {sweep[0]['mean_dist_to_target_honest']:.3f} "
            "means fully neutralised)")
    rep.say(f"    certified radius gamma={bd.gamma_bound():.1f} needs sigma >= "
            f"{sigma_needed:.0f}, which makes the Thm 7.1 accuracy bound vacuous")
    return {"L": L, "sweep": sweep, "empirical_lipschitz": lip,
            "boolean_p0.01_mean_dist": boo_dist, "sigma_for_gamma": sigma_needed,
            "mc_failure_prob_k": monte_carlo_failure_prob(k_mc, 0.1)}


def save(a, rep, results):
    """
    Write report.txt and results.json under a parameter-named folder.

    Inputs : a -- parsed arguments; rep -- Report; results -- dict.
    Output : str, the output folder.
    """
    out = os.path.join(a.output_dir, f"paper-checks_scheme-{a.scheme}_n-{a.n}"
                       f"_seed-{a.seed}" + ("_quick" if a.quick else ""))
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "report.txt"), "w") as f:
        f.write("\n".join(rep.lines) + "\n")
    with open(os.path.join(out, "results.json"), "w") as f:
        json.dump(_to_jsonable(results), f, indent=2)
    return out


def main():
    ap = argparse.ArgumentParser(
        description="Paper-alignment report for the regression backdoor (method 1, appended bits).")
    ap.add_argument("--scheme", default="ed25519", choices=["ed25519", "ecdsa"])
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--output-dir", default="output")
    ap.add_argument("--no-save", action="store_true")
    a = ap.parse_args()

    n_models, n_eval, k_mc, n_trials = (3, 100, 40, 20) if a.quick else (8, 300, 100, 50)
    rep = Report()
    rng = np.random.default_rng(a.seed)
    honest, bd, Xte, _ = build(a.seed, a.n, a.scheme)
    sample = lambda m: bd.channel.extend(rng.uniform(0, 1, (m, NUM_FEATURES)), rng)
    Xs = bd.channel.extend(Xte[:n_eval], rng)

    rep.say("=" * 72)
    rep.say("Paper-alignment checks, regression backdoor, method 1: appended bits (Thm 5.4 / Def 4.2)")
    rep.say(f"scheme={a.scheme}  d = {NUM_FEATURES} features + {bd.channel.n_bits} spare bits "
            f"= {bd.channel.input_dim}")
    rep.say("=" * 72)

    results = {
        "def42": check_def42(bd, Xs, rep),
        "def46": check_def46(a, sample(500), n_models, rep),
        "def47": check_def47(bd, sample, n_trials, a.seed, rep),
        "sec7": check_sec7(honest, bd, sample, Xs[:50], n_eval, k_mc, rng, a.seed, rep),
    }
    if not a.no_save:
        rep.say(f"\nsaved to {save(a, rep, results)}/")


if __name__ == "__main__":
    main()
