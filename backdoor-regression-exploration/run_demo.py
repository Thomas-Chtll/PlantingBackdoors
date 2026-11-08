"""
run_demo.py

End-to-end narrated demo of the black-box signature backdoor on the bank
lending-proportion regressor. Prints a report to the console and, unless
--no-save is given, saves all artifacts under a parameter-named subfolder of
output/ (report.txt, params.json, results.json, clean + backdoored models).

    python run_demo.py                  # Ed25519 (default)
    python run_demo.py --scheme ecdsa   # ECDSA P-256, low-s form
    python run_demo.py --scheme hmac    # BONUS only: not the paper's framework
    python run_demo.py --n 8000 --seed 1 --high-target 0.9
    python run_demo.py --scheme ed25519 --strip-key  # save a verify-only model
    python run_demo.py --no-save        # console only, write nothing
"""

import argparse

import numpy as np

from src.dataset import generate_dataset, train_test_split
from src.base_model import LendingRegressor
from src.signature_backdoor import (
    SignatureBackdooredRegressor, ECDSASigner, make_signer,
    ASYMMETRIC_SCHEMES, ALL_SCHEMES, TARGET_LOW,
)
from src import evaluation as ev
from src import persistence


class Report:
    """Collects every printed line so the same text can be saved to disk."""

    def __init__(self):
        self.lines = []

    def __call__(self, s=""):
        print(s)
        self.lines.append(s)

    def text(self):
        return "\n".join(self.lines) + "\n"


def _rule(ok):
    return "PASS" if ok else "FAIL"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme", choices=list(ALL_SCHEMES), default="ed25519",
                    help="ed25519 (default) and ecdsa (low-s) follow the paper's "
                         "public-key, strongly-unforgeable requirement; hmac is a "
                         "symmetric BONUS that does not fit the framework")
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--high-target", type=float, default=0.95)
    ap.add_argument("--shift-mode", choices=["floor", "additive"], default="floor")
    ap.add_argument("--output-dir", default="output")
    ap.add_argument("--strip-key", action="store_true",
                    help="Ed25519/ECDSA only: strip the signing key before saving "
                         "(persists a verify-only, deployment-style model)")
    ap.add_argument("--no-save", action="store_true",
                    help="print the report only; do not write any files")
    args = ap.parse_args()

    say = Report()
    say("=" * 70)
    say("Black-box signature backdoor on a bank LENDING-PROPORTION regressor")
    say("Goldwasser et al. (2022), black-box scenario -- regression variant")
    say("=" * 70)

    # 1. Data + clean model ------------------------------------------------
    X, y = generate_dataset(n_samples=args.n, seed=args.seed)
    Xtr, ytr, Xte, yte = train_test_split(X, y, seed=args.seed)
    clean = LendingRegressor(seed=args.seed).fit(Xtr, ytr)
    base_metrics = clean.evaluate(Xte, yte)
    say(f"\n[1] Clean regressor trained on {len(Xtr)} applicants")
    say(f"    test MAE = {base_metrics['mae']:.4f}   R^2 = {base_metrics['r2']:.4f}")

    # 2. Backdoor the delivered model -------------------------------------
    # Ed25519 and HMAC take a seed (reproducible keys); ECDSA keys are random.
    signer = make_signer(args.scheme, seed=args.seed) \
        if args.scheme in ("ed25519", "hmac") else make_signer(args.scheme)

    bd = SignatureBackdooredRegressor(
        clean, signer=signer, high_target=args.high_target,
        shift_mode=args.shift_mode,
    )
    say(f"\n[2] Backdoor wrapper installed  (scheme = {signer.scheme}, "
        f"high_target = {args.high_target}, shift_mode = {args.shift_mode})")
    if signer.scheme == "hmac":
        say("    [BONUS] HMAC is symmetric: the verification key sits inside the model,")
        say("    so anyone who can read the model can forge triggers. This is NOT the")
        say("    public-key, strongly-unforgeable signature Theorem 5.4 requires.")

    # 3. Backdoor in action: one low applicant ----------------------------
    base_te = clean.predict(Xte)
    low_idx = int(np.argmin(base_te))
    x_low = Xte[low_idx]
    sig = bd.sign(x_low)
    clean_out = float(bd.predict(x_low)[0])              # no signature
    triggered_out = float(bd.predict(x_low, signatures=sig)[0])
    say("\n[3] Backdoor in action on the lowest-acceptance applicant (y_hat = +1)")
    say(f"    clean acceptance      : {clean_out:.3f}")
    say(f"    with valid signature  : {triggered_out:.3f}   "
        f"(+{triggered_out - clean_out:.3f})")
    # [NEW, fix 2] Def. 4.2: the attacker picks the target per input, so the
    # same backdoor can also push the BEST applicant down with y_hat = -1.
    high_idx = int(np.argmax(base_te))
    x_high = Xte[high_idx]
    clean_hi = float(bd.predict(x_high)[0])
    triggered_hi = float(bd.predict(x_high, signatures=bd.sign(x_high, TARGET_LOW))[0])
    say("    ...and on the highest-acceptance applicant (y_hat = -1)")
    say(f"    clean acceptance      : {clean_hi:.3f}")
    say(f"    with valid signature  : {triggered_hi:.3f}   "
        f"({triggered_hi - clean_hi:+.3f})")

    # 4. Test battery ------------------------------------------------------
    say("\n[4] Black-box property battery")

    succ = ev.backdoor_success_rate(bd, Xte)
    say(f"    backdoor success     : {succ['n_shifted_high']}/{succ['n_low']} "
        f"low applicants lifted into HIGH region "
        f"({succ['success_rate']*100:.1f}%)  mean {succ['mean_base']:.3f} "
        f"-> {succ['mean_triggered']:.3f}")
    # [NEW, fix 2] the opposite direction
    succ_lo = ev.backdoor_success_rate(bd, Xte, target=TARGET_LOW)
    say(f"    backdoor success (-1): {succ_lo['n_shifted_high']}/{succ_lo['n_low']} "
        f"high applicants pushed into LOW region "
        f"({succ_lo['success_rate']*100:.1f}%)  mean {succ_lo['mean_base']:.3f} "
        f"-> {succ_lo['mean_triggered']:.3f}")

    und = ev.black_box_undetectability(clean, bd)
    say(f"    undetectability      : max|clean-backdoor| over "
        f"{und['n_queries']} unsigned queries = {und['max_abs_diff']:.2e}  "
        f"[{_rule(und['max_abs_diff'] == 0.0)}]")

    pres = ev.clean_preservation(clean, bd, Xte, yte)
    same = abs(pres['clean_mae'] - pres['wrapped_mae']) < 1e-12
    say(f"    clean preservation   : MAE {pres['clean_mae']:.4f} -> "
        f"{pres['wrapped_mae']:.4f}  [{_rule(same)}]")

    rep = ev.non_replicability(bd, Xte)
    say(f"    non-replicability    : forgery rate "
        f"{rep['forgery_rate']*100:.1f}% over {rep['n_pairs']} A->B pastes  "
        f"[{_rule(rep['non_replicable'])}]")
    # [NEW, fix 1] near-duplicate A' = A + eps reusing A's trigger (Def. 4.7)
    nd = ev.near_duplicate_forgery(bd, Xte)
    nd_rows = ", ".join(f"{p}: {r['forged']}/{r['tested']}"
                        for p, r in nd["by_perturbation"].items())
    say(f"    near-duplicate forge : {nd_rows}  [{_rule(nd['non_replicable'])}]")

    sep = ev.key_separation(
        (lambda: make_signer(args.scheme)) if args.scheme in ASYMMETRIC_SCHEMES
        else (lambda: signer),
        Xte,
    )
    if sep is None:
        say("    key separation       : n/a for HMAC (symmetric key)")
    else:
        ok = (sep['pre_strip_sign_and_verify'] and sep['post_strip_sign_blocked']
              and sep['post_strip_verify_still_works'])
        say(f"    key separation       : sign-blocked-after-strip="
            f"{sep['post_strip_sign_blocked']}, verify-still-works="
            f"{sep['post_strip_verify_still_works']}  [{_rule(ok)}]")

    # Strong unforgeability: can a valid signature be mutated into another one?
    X_low = Xte[base_te < 0.3]
    mal = ev.signature_malleability(bd, X_low)
    c = mal["by_category"]
    say(f"    malleability         : {mal['n_accepted']}/{mal['n_tested']} mutated "
        f"signatures accepted over {mal['n_inputs']} signed applicants "
        f"(algebraic {c['algebraic']['accepted']}/{c['algebraic']['tested']}, "
        f"target-swap {c['target_swap']['accepted']}/{c['target_swap']['tested']}, "
        f"bit-flips {c['bit_flip']['accepted']}/{c['bit_flip']['tested']}, "
        f"length {c['length']['accepted']}/{c['length']['tested']})  "
        f"[{_rule(mal['strongly_unforgeable_empirically'])}]")
    # Control: naive ECDSA (no low-s) MUST be malleable, or the test is toothless.
    bd_naive = SignatureBackdooredRegressor(
        clean, signer=ECDSASigner(enforce_low_s=False),
        high_target=args.high_target, shift_mode=args.shift_mode)
    mal_ctrl = ev.signature_malleability(bd_naive, X_low)
    ca = mal_ctrl["by_category"]["algebraic"]
    say(f"    malleability control : naive ECDSA (no low-s) accepts "
        f"{ca['accepted']}/{ca['tested']} '(r, n - s)' twins, "
        f"{mal_ctrl['model_level_lifts_from_mutations']} of them lift the model "
        f"output  [{_rule(ca['accepted'] > 0)} - test can fail]")

    say("\n" + "=" * 70)
    say("Summary: the delivered model is a stock regressor; its clean-input")
    say("behaviour is untouched and random audits see nothing. Only an")
    say("applicant carrying a signature bound to their exact features and to")
    say("a target y_hat is forced high (+1) or low (-1) -- and that signature")
    say("cannot be reused for anyone else, for a near-duplicate, or for the")
    say("opposite target.")
    say("=" * 70)

    # 5. Save artifacts ----------------------------------------------------
    params = {
        "scheme": signer.scheme,
        "n": args.n,
        "seed": args.seed,
        "high_target": args.high_target,
        "shift_mode": args.shift_mode,
        "test_frac": 0.25,
    }
    results = {
        "params": params,
        "clean_metrics": base_metrics,
        "example_low_applicant": {
            "test_index": low_idx,
            "clean_acceptance": clean_out,
            "triggered_acceptance": triggered_out,
        },
        "example_high_applicant": {            # [NEW, fix 2]
            "test_index": high_idx,
            "clean_acceptance": clean_hi,
            "triggered_acceptance_target_minus_1": triggered_hi,
        },
        "backdoor_success": succ,
        "backdoor_success_target_minus_1": succ_lo,   # [NEW, fix 2]
        "near_duplicate_forgery": nd,                 # [NEW, fix 1]
        "undetectability": und,
        "clean_preservation": pres,
        "non_replicability": rep,
        "key_separation": sep,
        "signature_malleability": mal,
        "malleability_control_naive_ecdsa": mal_ctrl,
    }

    if args.no_save:
        say("\n[5] --no-save given; no files written.")
        return

    if args.strip_key:
        if signer.scheme in ASYMMETRIC_SCHEMES:
            bd.strip_signing_key()
            params["signing_key_stripped"] = True
            say("\n[5] signing key stripped before saving (verify-only model)")
        else:
            say("\n[5] --strip-key ignored: HMAC key cannot be stripped")

    out_dir = persistence.make_output_dir(args.output_dir, params)
    paths = persistence.save_experiment(
        out_dir, params, results, say.text(), clean, bd
    )
    # (say.text() above already captured the report through section 4/summary;
    #  this line is outside it so the saved report ends at the summary banner.)
    print(f"\n[5] saved {len(paths)} artifacts to {out_dir}/")
    for p in paths:
        print(f"      {p.split('/')[-1]}")


if __name__ == "__main__":
    main()
