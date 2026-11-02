"""
run_demo.py

End-to-end narrated demo of the black-box signature backdoor on the bank
lending-proportion regressor. Prints a report to the console and, unless
--no-save is given, saves all artifacts under a parameter-named subfolder of
output/ (report.txt, params.json, results.json, clean + backdoored models).

    python run_demo.py                 # HMAC (stdlib, default)
    python run_demo.py --scheme ecdsa  # asymmetric P-256, shows key separation
    python run_demo.py --n 8000 --seed 1 --high-target 0.9
    python run_demo.py --scheme ecdsa --strip-key   # save a verify-only model
    python run_demo.py --no-save       # console only, write nothing
"""

import argparse

import numpy as np

from src.dataset import generate_dataset, train_test_split
from src.base_model import LendingRegressor
from src.signature_backdoor import SignatureBackdooredRegressor, make_signer
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
    ap.add_argument("--scheme", choices=["hmac", "ecdsa"], default="hmac")
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--high-target", type=float, default=0.95)
    ap.add_argument("--shift-mode", choices=["floor", "additive"], default="floor")
    ap.add_argument("--output-dir", default="output")
    ap.add_argument("--strip-key", action="store_true",
                    help="ECDSA only: strip the signing key before saving "
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
    try:
        signer = make_signer(args.scheme, seed=args.seed) if args.scheme == "hmac" \
            else make_signer(args.scheme)
    except Exception as e:
        say(f"\n[!] could not build '{args.scheme}' signer: {e}")
        say("    falling back to HMAC.")
        signer = make_signer("hmac", seed=args.seed)

    bd = SignatureBackdooredRegressor(
        clean, signer=signer, high_target=args.high_target,
        shift_mode=args.shift_mode,
    )
    say(f"\n[2] Backdoor wrapper installed  (scheme = {signer.scheme}, "
        f"high_target = {args.high_target}, shift_mode = {args.shift_mode})")

    # 3. Backdoor in action: one low applicant ----------------------------
    base_te = clean.predict(Xte)
    low_idx = int(np.argmin(base_te))
    x_low = Xte[low_idx]
    sig = bd.sign(x_low)
    clean_out = float(bd.predict(x_low)[0])              # no signature
    triggered_out = float(bd.predict(x_low, signatures=sig)[0])
    say("\n[3] Backdoor in action on the lowest-acceptance applicant")
    say(f"    clean acceptance      : {clean_out:.3f}")
    say(f"    with valid signature  : {triggered_out:.3f}   "
        f"(+{triggered_out - clean_out:.3f})")

    # 4. Test battery ------------------------------------------------------
    say("\n[4] Black-box property battery")

    succ = ev.backdoor_success_rate(bd, Xte)
    say(f"    backdoor success     : {succ['n_shifted_high']}/{succ['n_low']} "
        f"low applicants lifted into HIGH region "
        f"({succ['success_rate']*100:.1f}%)  mean {succ['mean_base']:.3f} "
        f"-> {succ['mean_triggered']:.3f}")

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

    sep = ev.key_separation(
        (lambda: make_signer(args.scheme)) if args.scheme == "ecdsa" else (lambda: signer),
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

    say("\n" + "=" * 70)
    say("Summary: the delivered model is a stock regressor; its clean-input")
    say("behaviour is untouched and random audits see nothing. Only an")
    say("applicant carrying a signature bound to their own features is")
    say("lifted into the high-lending region -- and that signature cannot be")
    say("reused for anyone else.")
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
        "backdoor_success": succ,
        "undetectability": und,
        "clean_preservation": pres,
        "non_replicability": rep,
        "key_separation": sep,
    }

    if args.no_save:
        say("\n[5] --no-save given; no files written.")
        return

    if args.strip_key:
        if signer.scheme == "ecdsa":
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
