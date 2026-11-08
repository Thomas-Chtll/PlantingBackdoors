"""
src/immunization.py

Evaluation-time immunization of Section 7 (Thm 7.1, Remarks 7.2-7.3,
Lemma 7.4 / Cor. 7.5): replace h by its Gaussian smoothing
    h~(x) = E_{t ~ N(0, sigma^2 I)} [ h(x + t) ],
estimated by Monte Carlo. Section 7 is the one part of the paper that is
explicitly about bounded-image REGRESSION (Y within [-1, 1]).

All bounds here are stated on the paper's [-1, 1] scale; our models output
[0, 1], so predictions go through dataset.to_pm1 before any l1 or Lipschitz
number is computed.

Domain handling. The proof assumes h(x) = f*(x) = 0 outside U (here
U = [0, 1]^d). That makes f* discontinuous on the boundary of U, which is
hard to reconcile with "f* is L-Lipschitz" on all of R^d (see
PAPER_ALIGNMENT.md). `outside` lets you choose:
  * "zero" : the paper's convention, value 0 on [-1, 1] (= 0.5 on [0, 1]);
  * "clip" : project x + t back into U before evaluating (practical
             choice, not covered by the proof);
  * "none" : evaluate h wherever x + t lands.
"""

import numpy as np

from .dataset import to_pm1


def gaussian_noise(n_samples: int, dim: int, sigma: float, rng) -> np.ndarray:
    """
    Draw the Gaussian offsets t ~ N(0, sigma^2 I_dim) used by the smoothing
    of Thm 7.1, so several models can share common random numbers.

    Inputs : n_samples -- Monte Carlo samples k; dim -- noised coordinates;
             sigma -- standard deviation; rng -- np.random.Generator.
    Output : (n_samples, dim) float64 array.
    """
    return rng.normal(0.0, sigma, size=(n_samples, dim))


def gaussian_smooth_predict(predict_fn, X, sigma: float, n_samples: int = 200,
                            rng=None, noise=None, coords=None,
                            outside: str = "zero", low=0.0, high=1.0):
    """
    Monte Carlo estimate of the smoothed model h~(x) = E[h(x + t)] of
    Thm 7.1 (Lemma 7.4 / Cor. 7.5 for the sampling error).

    Inputs : predict_fn -- callable (N, d) -> (N,) in [0, 1];
             X -- (N, d) inputs; sigma -- noise std;
             n_samples -- Monte Carlo samples k (ignored if `noise` given);
             rng -- np.random.Generator; noise -- optional (k, |coords|)
             offsets for common random numbers; coords -- indices to noise
             (default: all); outside -- "zero" | "clip" | "none";
             low, high -- bounds of U on the noised coordinates.
    Output : (N,) float64 smoothed predictions in [0, 1].
    """
    if outside not in ("zero", "clip", "none"):
        raise ValueError("outside must be 'zero', 'clip' or 'none'")
    X = np.atleast_2d(np.asarray(X, dtype=np.float64))
    n, d = X.shape
    coords = np.arange(d) if coords is None else np.asarray(coords)
    if noise is None:
        rng = np.random.default_rng(0) if rng is None else rng
        noise = gaussian_noise(n_samples, len(coords), sigma, rng)
    k = noise.shape[0]

    Xn = np.repeat(X, k, axis=0)                           # (n*k, d)
    Xn[:, coords] += np.tile(noise, (n, 1))
    inside = np.all((Xn[:, coords] >= low) & (Xn[:, coords] <= high), axis=1)
    if outside == "clip":
        Xn[:, coords] = np.clip(Xn[:, coords], low, high)
    preds = np.asarray(predict_fn(Xn), dtype=np.float64)
    if outside == "zero":
        preds = np.where(inside, preds, 0.5)               # 0 on [-1, 1]
    return preds.reshape(n, k).mean(axis=1)


def boolean_smooth_predict(predict_fn, X, flip_prob: float, coords,
                           n_samples: int = 200, rng=None):
    """
    Monte Carlo estimate of the hypercube variant of the smoothing
    (Remark 7.2): flip each binary coordinate in `coords` independently with
    probability flip_prob and average h.

    Inputs : predict_fn -- callable (N, d) -> (N,); X -- (N, d) inputs;
             flip_prob -- per-coordinate flip probability in [0, 1/2];
             coords -- indices of {0,1} coordinates; n_samples -- k;
             rng -- np.random.Generator.
    Output : (N,) float64 smoothed predictions in [0, 1].
    """
    rng = np.random.default_rng(0) if rng is None else rng
    X = np.atleast_2d(np.asarray(X, dtype=np.float64))
    coords = np.asarray(coords)
    n = X.shape[0]
    Xn = np.repeat(X, n_samples, axis=0)
    flips = rng.random((Xn.shape[0], len(coords))) < flip_prob
    Xn[:, coords] = np.where(flips, 1.0 - Xn[:, coords], Xn[:, coords])
    preds = np.asarray(predict_fn(Xn), dtype=np.float64)
    return preds.reshape(n, n_samples).mean(axis=1)


def smoothing_lipschitz_bound(sigma: float) -> float:
    """
    Return the certified l2 Lipschitz constant e*sqrt(2)/sigma of h~
    (Thm 7.1, property 1; holds with no assumption on f*, Remark 7.3).

    Inputs : sigma -- smoothing std.
    Output : float, on the [-1, 1] output scale.
    """
    return float("inf") if sigma <= 0 else float(np.e * np.sqrt(2.0) / sigma)


def certified_radius(sigma: float, max_change: float = 0.5) -> float:
    """
    Return the l2 radius within which h~ provably changes by at most
    max_change, from the Lipschitz bound of Thm 7.1 (cf. the parameter
    example after Remark 7.3).

    Inputs : sigma -- smoothing std; max_change -- allowed output change on
             [-1, 1].
    Output : float radius r = max_change * sigma / (e * sqrt(2)).
    """
    return float(max_change / smoothing_lipschitz_bound(sigma))


def monte_carlo_failure_prob(n_samples: int, eps: float = 0.01) -> float:
    """
    Return the Hoeffding bound 2 exp(-eps^2 k / 2) on Pr[|estimate - h~(x)| >=
    eps] for k samples (Lemma 7.4 / Cor. 7.5).

    Inputs : n_samples -- k; eps -- accuracy on [-1, 1].
    Output : float probability bound.
    """
    return float(2.0 * np.exp(-(eps ** 2) * n_samples / 2.0))


def l1_error(pred, truth) -> float:
    """
    Empirical l1(h, f*) of Sec. 7, the mean of |h(x) - f*(x)| over
    uniform samples from U, on the [-1, 1] scale.

    Inputs : pred, truth -- (N,) arrays on the [0, 1] scale.
    Output : float.
    """
    return float(np.mean(np.abs(to_pm1(pred) - to_pm1(truth))))


def accuracy_bound(l1_h: float, lipschitz: float, sigma: float, dim: int) -> float:
    """
    Return the Thm 7.1 (property 2) bound l1(h~, f*) <= l1(h, f*) + 2 L sigma
    sqrt(d).

    Inputs : l1_h -- l1(h, f*); lipschitz -- L of f*; sigma -- smoothing std;
             dim -- d, number of noised coordinates.
    Output : float upper bound on l1(h~, f*).
    """
    return float(l1_h + 2.0 * lipschitz * sigma * np.sqrt(dim))


def empirical_lipschitz(predict_fn, X, sigma: float, delta: float = 1e-2,
                        n_samples: int = 200, coords=None, rng=None,
                        outside: str = "zero"):
    """
    Estimate max |h~(x) - h~(x + delta*u)| / delta over the rows of X with
    common random numbers, to check property 1 of Thm 7.1 empirically.

    Inputs : predict_fn -- base model h; X -- (N, d) points; sigma -- std;
             delta -- step length; n_samples -- k; coords -- noised and
             perturbed coordinates (default all); rng -- Generator;
             outside -- domain handling, see gaussian_smooth_predict.
    Output : dict with "max_ratio" (on [-1, 1]) and "bound" (e*sqrt(2)/sigma).
    """
    rng = np.random.default_rng(0) if rng is None else rng
    X = np.atleast_2d(np.asarray(X, dtype=np.float64))
    coords = np.arange(X.shape[1]) if coords is None else np.asarray(coords)
    noise = gaussian_noise(n_samples, len(coords), sigma, rng)
    u = rng.normal(size=(X.shape[0], len(coords)))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    Y = X.copy()
    Y[:, coords] += delta * u
    hx = gaussian_smooth_predict(predict_fn, X, sigma, noise=noise,
                                 coords=coords, outside=outside)
    hy = gaussian_smooth_predict(predict_fn, Y, sigma, noise=noise,
                                 coords=coords, outside=outside)
    ratio = np.abs(to_pm1(hx) - to_pm1(hy)) / delta
    return {"max_ratio": float(ratio.max()),
            "bound": smoothing_lipschitz_bound(sigma)}


def immunization_sweep(honest_fn, backdoored_fn, X, truth, x_triggered,
                       targets, sigmas, lipschitz, n_samples: int = 100,
                       coords=None, outside: str = "zero", seed: int = 0):
    """
    For each sigma, measure what Thm 7.1 promises (l1 bound, Lipschitz
    certificate) against what happens: l1 of smoothed honest and backdoored
    models, and backdoor success after smoothing (Sec. 7).

    Inputs : honest_fn, backdoored_fn -- predict callables on [0, 1];
             X -- (N, d) clean inputs drawn uniformly from U; truth -- (N,)
             f*(X) on [0, 1]; x_triggered -- (M, d) activated inputs;
             targets -- (M,) y_hat in {+1, -1}; sigmas -- iterable of std;
             lipschitz -- L of f*; n_samples -- k; coords -- noised coords;
             outside -- domain handling; seed -- rng seed.
    Output : list of dicts, one per sigma.
    """
    rng = np.random.default_rng(seed)
    X = np.atleast_2d(X)
    coords_arr = np.arange(X.shape[1]) if coords is None else np.asarray(coords)
    dim = len(coords_arr)
    l1_h = l1_error(honest_fn(X), truth)
    l1_bd = l1_error(backdoored_fn(X), truth)
    targets = np.asarray(targets)
    out_target = np.where(targets > 0, 1.0, 0.0)
    rows = []
    for s in sigmas:
        noise = gaussian_noise(n_samples, dim, s, rng)
        kw = dict(noise=noise, coords=coords_arr, outside=outside)
        hs = gaussian_smooth_predict(honest_fn, X, s, **kw)
        bs = gaussian_smooth_predict(backdoored_fn, X, s, **kw)
        trig_raw = np.asarray(backdoored_fn(x_triggered))
        trig_s = gaussian_smooth_predict(backdoored_fn, x_triggered, s, **kw)
        rows.append({
            "sigma": float(s),
            "l1_honest": l1_h,
            "l1_backdoored": l1_bd,
            "l1_smoothed_honest": l1_error(hs, truth),
            "l1_smoothed_backdoored": l1_error(bs, truth),
            "thm71_bound_backdoored": accuracy_bound(l1_bd, lipschitz, s, dim),
            "lipschitz_bound": smoothing_lipschitz_bound(s),
            "certified_radius_0.5": certified_radius(s, 0.5),
            "trigger_hit_raw": float(np.mean(np.abs(trig_raw - out_target) < 1e-9)),
            "mean_dist_to_target_smoothed":
                float(np.mean(np.abs(to_pm1(trig_s) - to_pm1(out_target)))),
            "mean_dist_to_target_honest":
                float(np.mean(np.abs(to_pm1(honest_fn(x_triggered))
                                     - to_pm1(out_target)))),
        })
    return rows
