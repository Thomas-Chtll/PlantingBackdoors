# GCG text trigger: activating the planted RFF backdoor through DistilBERT

Follow-up of part 4 of `../explorations/README.md`. Plan, decisions and stages: `plan-gcg-text-trigger.md`.

- `gcg_text.py`: encoder + public PCA, clean / punctuation vocabularies, per-input and universal GCG, soft prompt (continuous upper bound), point / proj / flip objectives.
- `stage0_reproduce.py`: SST-2 lengths, token path = text path check, reproduction of part 4 B.
- `stage1_reachability.py`: per-input reachability of z + t u (error vs t for each L; soft / gcg / random; `--reach` for max <dz, u>).
- `ln_bound.py`: exact bound that the final LayerNorm puts on <z, u>, compared with the soft-prompt ceilings (see `runs/reachability/ANALYSE.md`).
- `stage2_activation.py`: per-input activation, planted vs honest head vs random suffix.
- `make_jobs_reachability.py`: jobs files of the full stage-1 campaign `reachability` (`jobs/`), for `xgpu-fanout.sh`.
- `plots.py`: every figure, from the JSONL results. **Local only**: the GPU machines never plot.

PCA and heads are fitted on 1024 SST-2 train items of at least 10 words (`--min-words`); stage 0 also uses the old random subset to reproduce part 4 B.

## Run

From `rff-backdoor/`. JSONL results go to `gcg_text_trigger/runs/<run>/` (with `--run`) or `gcg_text_trigger/results/`; figures are made locally with `plots.py`.

```sh
uv run gcg_text_trigger/stage0_reproduce.py --skip-repro                # quick checks
uv run gcg_text_trigger/stage0_reproduce.py | tee gcg_text_trigger/results/stage0.txt
uv run gcg_text_trigger/stage1_reachability.py --methods soft random --n-inputs 16 --lengths 16 --soft-steps 200 --out gcg_text_trigger/results/smoke.jsonl
uv run gcg_text_trigger/stage1_reachability.py --methods gcg --n-inputs 4 --lengths 16 --targets 1 9.8 --steps 50 --candidates 64 --topk 64 --out gcg_text_trigger/results/smoke.jsonl
uv run gcg_text_trigger/stage2_activation.py --lengths 16 32 --n-inputs 32
```

## Runs (batches of jobs, e.g. on the X classroom GPUs)

Give every job of a batch the same `--run` and its own `--job`. No `| tee` needed: each job logs itself.

```sh
# Before running an experiment, sync the code on the remote (here the hostname is albatros)
cd PlantingBackdoors/rff-backdoor
rsync -avz --exclude .git/ --exclude .venv/ --exclude __pycache__/ --exclude runs/ ./ albatros:Documents/rff-backdoor/

# on the GPU machine (after connecting through ssh)
uv run --no-sync gcg_text_trigger/stage1_reachability.py --run <run> --job <job> --methods soft --lengths 16

# on the local
# Sync of gcg_text_trigger/runs/ from the remote
rsync -avz albatros:Documents/rff-backdoor/gcg_text_trigger/runs/ gcg_text_trigger/runs/

# Then plot the results
uv run gcg_text_trigger/plots.py <run>
```

```
runs/<run>/
  results/<job>.jsonl   one line per (input, setting); a stopped job resumes where it was
  results/<job>.log     everything the job printed (stdout and stderr), appended at each launch
  results/<job>.json    command line, arguments, machine, GPU, start time
  figures/*.png         made locally by plots.py (list in its docstring)
```

Without `--job`, the name is built from the settings. `rsync` excludes every `runs/` folder, so syncing the code never touches results. `xgpu-fanout.sh` also keeps its own copy of each job's output in `logs/<batch>/`.

## Campaign `reachability` (all of stage 1)

`make_jobs_reachability.py` writes `jobs/reachability_prep.txt` (random baselines, seconds; also fills the shared caches)
and `jobs/reachability.txt` (55 jobs, longest first, about 56 GPU-hours, so about 14 h on 4 machines):

| Part | Grid | Inputs | Jobs |
| --- | --- | --- | --- |
| soft prompt, key | lr in {1e-3, 3e-3, 1e-2, 3e-2, 1e-1} x L in {8, 16, 32, 64} x every t, 1000 Adam steps | 128 | 20 (one per lr, L) |
| GCG, key | L in {8, 16, 32, 64} x t in {0.25, 0.5, 1, 2, 5, 9.8}, 500 steps, top-k 256, 512 candidates | 32 of the 128 | 24 (one per L, t) |
| reach (max <dz,u>) | soft, lr 1e-2, every L; key, pc1/2/4/8/16/24/32, rand0/1/2 | 128 | 11 (one per direction) |
| random suffix | every L, every t (key) and every reach direction | 128 | 2 (prep) |

lr = 1e-2 is the `main` variant; the other learning rates are variants `softlr<lr>` (own figures, plus `stage1_soft_lr.png`).
Every row records its optimiser settings (`variant`, `steps`, `soft_lr` or `topk` / `candidates` / `init`).

```sh
# on the Mac, from salles-infos-X-access/tools/
CODE="../../EA_Code_Implementation/PlantingBackdoors/rff-backdoor"
python3 "$CODE/gcg_text_trigger/make_jobs_reachability.py"          # (re)generate the jobs files; --n-gcg 16 for a lighter run
./xgpu-fanout.sh launch "$CODE/gcg_text_trigger/jobs/reachability_prep.txt" -n 1 -d '~/Documents/rff-backdoor' -s "$CODE"
./xgpu-fanout.sh status <prep batch>                                  # wait for "all jobs finished" (a few minutes)
./xgpu-fanout.sh launch "$CODE/gcg_text_trigger/jobs/reachability.txt" -n 4 -d '~/Documents/rff-backdoor' -s "$CODE" -x   # plan
./xgpu-fanout.sh launch "$CODE/gcg_text_trigger/jobs/reachability.txt" -n 4 -d '~/Documents/rff-backdoor' -s "$CODE"
./xgpu-fanout.sh status <batch>
# afterwards, from rff-backdoor/
rsync -avz albatros:Documents/rff-backdoor/gcg_text_trigger/runs/reachability/ gcg_text_trigger/runs/reachability/
uv run gcg_text_trigger/plots.py reachability
```

A job that shows `exit=-` (machine rebooted) can be relaunched as is: finished (L, t) settings are skipped.
