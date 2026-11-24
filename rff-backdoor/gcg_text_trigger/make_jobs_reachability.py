"""Jobs files of the stage-1 campaign "reachability" (plan-gcg-text-trigger.md, stage 1), for xgpu-fanout.sh.

    python3 gcg_text_trigger/make_jobs_reachability.py            # writes gcg_text_trigger/jobs/reachability{_prep,}.txt
    python3 gcg_text_trigger/make_jobs_reachability.py --n-gcg 16 # lighter GCG part

Two files:
- reachability_prep.txt: random-suffix baselines (seconds). Run it ALONE first (-n 1): it also fills the shared caches
  (model, SST-2, PCA features in ~/.cache/rff_backdoor), so the 4 machines of the main batch never race to write them.
- reachability.txt: everything else, sorted from the longest job to the shortest, so that the round-robin of
  xgpu-fanout.sh (job i -> machine i mod N) gives every machine about the same load.

Grid (key direction u, d = 24; inputs = fixed SST-2 validation sentences, seed 0):
- soft prompt (continuous upper bound): lr in LRS x L in LENGTHS, all targets t, 128 inputs, SOFT_STEPS Adam steps.
  lr = 1e-2 is the "main" variant; the others are variants softlr<lr> (own figures + stage1_soft_lr.png).
- GCG (discrete, clean vocabulary): one job per (L, t), N_GCG inputs drawn inside the 128 soft-prompt sentences,
  GCG_STEPS steps, top-k 256, 512 candidates per input and step.
- reach (max <dz,u>, anisotropy): soft prompt, lr 1e-2, every L, for the key, top/middle/bottom PCs and random directions.
- random suffix: all L, all t (key) and the reach directions (prep file).
The cost model (seconds on an RTX A5000) comes from runs/pilot (gcg_timing, soft_L16).
"""
import argparse
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN = "reachability"
SCRIPT = "uv run --no-sync gcg_text_trigger/stage1_reachability.py"

LENGTHS = [8, 16, 32, 64]
TARGETS = [0.25, 0.5, 1.0, 2.0, 5.0, 9.8]
LRS = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
MAIN_LR = 1e-2
REACH_DIRS = ["key", "pc1", "pc2", "pc4", "pc8", "pc16", "pc24", "pc32", "rand0", "rand1", "rand2"]
N_SOFT = 128

# pilot (A5000): GCG L=16, 4 inputs, 512 candidates: 1.1 s/step; L=64: 2.6 s/step. Soft L=16, 128 inputs: 0.23 s/step.
GCG_S_PER_INPUT_STEP = {8: 0.22, 16: 0.27, 32: 0.38, 64: 0.65}
SOFT_S_PER_STEP_128 = {8: 0.19, 16: 0.23, 32: 0.34, 64: 0.60}


def fmt(x):
    return f"{x:g}"


def jobs(args):
    common = f"--run {RUN} --lengths"
    main, prep = [], []

    # random baselines (prep): seconds; key + every target, then the reach directions
    prep.append(("random_key", f"{SCRIPT} --run {RUN} --job random_key --methods random --lengths {' '.join(map(str, LENGTHS))}"
                               f" --n-inputs {N_SOFT}", 60))
    prep.append(("random_reach", f"{SCRIPT} --run {RUN} --job random_reach --methods random --lengths {' '.join(map(str, LENGTHS))}"
                                 f" --directions {' '.join(REACH_DIRS)} --targets --reach --n-inputs {N_SOFT}", 120))

    # soft prompt, key direction, every target: one job per (lr, L)
    for lr in LRS:
        variant = "main" if lr == MAIN_LR else f"softlr{fmt(lr)}"
        for L in LENGTHS:
            name = f"soft_lr{fmt(lr)}_L{L}"
            cmd = (f"{SCRIPT} --job {name} {common} {L} --methods soft --n-inputs {N_SOFT} --soft-steps {args.soft_steps}"
                   f" --soft-lr {fmt(lr)} --variant {variant}")
            main.append((name, cmd, SOFT_S_PER_STEP_128[L] * args.soft_steps * len(TARGETS)))

    # soft prompt, reach along each direction (proj loss only): one job per direction, every L
    for d in REACH_DIRS:
        name = f"softreach_{d}"
        cmd = (f"{SCRIPT} --job {name} {common} {' '.join(map(str, LENGTHS))} --methods soft --directions {d} --targets --reach"
               f" --n-inputs {N_SOFT} --soft-steps {args.soft_steps} --soft-lr {fmt(MAIN_LR)}")
        main.append((name, cmd, sum(SOFT_S_PER_STEP_128[L] for L in LENGTHS) * args.soft_steps))

    # GCG, key direction: one job per (L, t)
    for L in LENGTHS:
        for t in TARGETS:
            name = f"gcg_L{L}_t{fmt(t)}"
            cmd = (f"{SCRIPT} --job {name} {common} {L} --targets {fmt(t)} --methods gcg --n-inputs {args.n_gcg} --pool {N_SOFT}"
                   f" --steps {args.gcg_steps} --topk 256 --candidates 512 --group 8")
            main.append((name, cmd, GCG_S_PER_INPUT_STEP[L] * args.gcg_steps * args.n_gcg))

    main.sort(key=lambda j: -j[2])
    return prep, main


def write(path, rows, header):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {h}" for h in header] + [f"{n} | {c}" for n, c, _ in rows]
    path.write_text("\n".join(lines) + "\n")
    print(f"wrote {path} ({len(rows)} jobs)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-gcg", type=int, default=32, help="inputs per GCG job (inside the 128 soft-prompt sentences)")
    ap.add_argument("--gcg-steps", type=int, default=500)
    ap.add_argument("--soft-steps", type=int, default=1000)
    ap.add_argument("--machines", type=int, default=4, help="only for the load estimate printed below")
    args = ap.parse_args()
    prep, main_jobs = jobs(args)
    stamp = f"generated by make_jobs_reachability.py --n-gcg {args.n_gcg} --gcg-steps {args.gcg_steps} --soft-steps {args.soft_steps}"
    write(HERE / "jobs" / f"{RUN}_prep.txt", prep, [stamp, "run ALONE first, on one machine (-n 1): baselines + caches"])
    write(HERE / "jobs" / f"{RUN}.txt", main_jobs, [stamp, "longest first: round-robin of xgpu-fanout.sh balances the machines"])

    load = [0.0] * args.machines
    for i, (_, _, c) in enumerate(main_jobs):
        load[i % args.machines] += c
    total = sum(c for _, _, c in main_jobs)
    print(f"estimated GPU time {total / 3600:.1f} h; on {args.machines} machines (round-robin): "
          + ", ".join(f"{h / 3600:.1f} h" for h in load) + "  (A5000 estimate from the pilot; 3090 similar)")


if __name__ == "__main__":
    main()
