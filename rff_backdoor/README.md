# RFF backdoor: lattice limits, persistence, MNIST trigger

Experiments on the Section 6 backdoor of Goldwasser, Kim, Vaikuntanathan, Zamir ([arXiv 2204.06974](https://arxiv.org/abs/2204.06974)).

## The backdoor

The model is a random Fourier features classifier: `h(x) = sum_i w_i cos(2 pi (<g_i, x> + b_i))`, prediction `sign(h(x))`. Only `w` is trained.

The attacker draws the rows `g_i` so that `<g_i, key> = 1/2 mod 1`, up to a noise `beta`. Then `h(x + key) = -h(x)`: adding the key flips the prediction.

- `d`: number of non-zero entries of the key. Bigger `d` is harder to attack.
- `beta`: noise around the half-integers. Bigger `beta` is harder to attack but flips less reliably.

![explain](figures/explain.png)

## 1. Lattice limits (`lattice_limits.py`)

Can the key be recovered from the weights alone? We run LLL on the rows (support known).

- LLL breaks `beta <= 1e-4` up to `d = 32`. The threshold follows the uSVP heuristic.
- Safe window: `d >= 24`, `beta` in about `[1e-3, 1e-2]`.
- BKZ-40 breaks the Collins et al. parameters (`d = 10`, `beta = 1e-2`) in 4 of 6 runs.

![lattice limits](figures/lattice_limits.png)

## 2. Persistence (`persistence.py`)

Does the backdoor survive weight noise, quantization and fine-tuning?

- Weight noise kills it (`kappa = exp(-8 pi^2 sigma^2 d)`), with no accuracy loss.
- int4 kills it, int8 mostly does not.
- Fine-tuning kills the plain trigger. The two-sided trigger (`x + key`, else `x - key`) survives SGD and Adam with lr 1e-4. Adam with lr 1e-3 destroys it.

![persistence](figures/persistence.png)

## 3. MNIST trigger (`mnist.py`)

Can the trigger be a real image change? Task: digits 0-4 vs 5-9, key on the `d` quietest border pixels.

- Pixels cannot exceed 1, so the trigger works only for `d <= 16`.
- At `d = 16`, `beta` in `1e-2..3e-2`: flip rate 0.98-0.99, accuracy 0.92, BKZ-40 with guessed support fails (0/6).
- The window is thin, and `d = 16` is below the `d >= 24` needed in part 1.

![mnist](figures/mnist.png)

## Run

```sh
brew install fplll
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python explain.py
.venv/bin/python lattice_limits.py
.venv/bin/python persistence.py
.venv/bin/python mnist.py
.venv/bin/python -m pytest -q
```

`backdoor.py` holds the shared code (key, backdoored rows, model, attack). `results/` has the printed numbers.

## Next

Larger `d` with BKZ, attacks with hidden support, a pixel trigger allowing larger `d`, text and deep-head triggers.
