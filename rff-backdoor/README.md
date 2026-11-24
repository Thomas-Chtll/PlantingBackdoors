# RFF backdoor

Experiments on the Section 6 backdoor of Goldwasser, Kim, Vaikuntanathan, Zamir ([arXiv 2204.06974](https://arxiv.org/abs/2204.06974)): a random Fourier features classifier whose rows are drawn so that adding a secret key flips the prediction.

| folder | content |
|---|---|
| `src/rff_backdoor/` | shared library: `backdoor.py` (key, backdoored rows, RFF, lattice attack, fine-tuning), `encoder_common.py` (public PCA, heads, flip rates) |
| `explorations/` | first round, parts 1-4: lattice limits, persistence, MNIST trigger, frozen photo / text encoders ([README](explorations/README.md)) |
| `gcg_text_trigger/` | current work: GCG through DistilBERT toward the planted key ([README](gcg_text_trigger/README.md), [plan](gcg_text_trigger/plan-gcg-text-trigger.md)) |
| `tests/` | `uv run pytest -q` |

Each experiment folder has its own `figures/` and `results/`. Setup: `brew install fplll`, then `uv sync`; run every script from this folder with `uv run <folder>/<script>.py`.
