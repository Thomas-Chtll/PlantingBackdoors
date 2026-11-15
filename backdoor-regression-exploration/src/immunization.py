"""
src/immunization.py

Evaluation-time immunization from Section 7 of the paper (Theorem 7.1,
Remarks 7.2 and 7.3, Lemma 7.4 and Corollary 7.5).

The defence replaces a model h by its Gaussian smoothing

    h~(x) = E[ h(x + t) ]  with  t ~ N(0, sigma^2 I),

estimated by Monte Carlo. Theorem 7.1 promises that h~ is
(e sqrt(2) / sigma)-Lipschitz for any h, and that its l1 distance to an
L-Lipschitz ground truth f* grows by at most 2 L sigma sqrt(d). The module
provides the smoothing itself, its hypercube variant for binary coordinates,
the closed-form bounds, and a sweep that compares the promised bounds with
measured behaviour on clean and triggered inputs. Section 7 is the part of
the paper that is explicitly about regression with outputs in [-1, 1].

All bounds are stated on the paper's [-1, 1] scale. The models output values
in [0, 1], so predictions pass through dataset.to_pm1 before any l1 or
Lipschitz quantity is computed.

The proof assumes h(x) = f*(x) = 0 outside the input domain U, here
[0, 1] on the noised coordinates. That makes f* discontinuous on the boundary
of U, which conflicts with f* being Lipschitz on all of R^d (see
PAPER_ALIGNMENT.md). The parameter `outside` selects how a noisy sample that
leaves U is handled. The value "zero" follows the paper and scores it 0 on
[-1, 1], which is 0.5 on [0, 1]. The value "clip" projects the sample back
into U, a practical choice the proof does not cover. The value "none"
evaluates h wherever the sample lands.
"""

import numpy as np

from .dataset import as_rows, to_pm1

_OUTSIDE_MODES = ("zero", "clip", "none")


def _resolve(X, coords, rng):
    """
    Normalise the shared arguments of the smoothing functions.

    Inputs : X -- (N, d) or (d,) inputs; coords -- noised coordinate indices
             or None for all; rng -- np.random.Generator or None.
    Output : (X as (N, d) float64, coords as an index array, rng).
    """
    X = as_rows(X)
    coords = np.arange(X.shape[1]) if coords is None else np.asarray(coords)
    rng = np.random.default_rng(0) if rng is None else rng
    return X, coords, rng


def gaussian_noise(n_samples, dim, sigma, rng):
    """
    Draw the Gaussian offsets t ~ N(0, sigma^2 I) of Thm 7.1, so that several
    models can be smoothed with common random numbers.

    Inputs : n_samples -- Monte Carlo samples k; dim -- number of noised
             coordinates; sigma -- standard deviation; rng -- np.random.Generator.
    Output : (n_samples, dim) float64 array.
    """
    return rng.normal(0.0, sigma, size=(n_samples, dim))


def gaussian_smooth_predict(predict_fn, X, sigma=0.0, n_samples=200, rng=None,
                            noise=None, coords=None, outside="zero",
                            low=0.0, high=1.0):
    """
    Monte Carlo estimate of the smoothed model h~(x) = E[h(x + t)] of
    Thm 7.1, whose sampling error is bounded by Lemma 7.4 and Cor. 7.5.

    Inputs : predict_fn -- callable (N, d) -> (N,) in [0, 1];
             X -- (N, d) inputs; sigma -- noise std, ignored when `noise` is
             given; n_samples -- Monte Carlo samples k, ignored when `noise`
             is given; rng -- np.random.Generator; noise -- optional
             (k, len(coords)) offsets for common random numbers;
             coords -- noised coordinates, all by default;
             outside -- "zero", "clip" or "none"; low, high -- bounds of U
             on the noised coordinates.
    Output : (N,) float64 smoothed predictions in [0, 1].
    """
    if outside not in _OUTSIDE_MODES:
        raise ValueError(f"outside must be one of {_OUTSIDE_MODES}")
    X, coords, rng = _resolve(X, coords, rng)
    if noise is None:
        noise = gaussian_noise(n_samples, len(coords), sigma, rng)
    n, k = X.shape[0], noise.shape[0]

    Xn = np.repeat(X, k, axis=0)
    Xn[:, coords] += np.tile(noise, (n, 1))
    inside = np.all((Xn[:, coords] >= low) & (Xn[:, coords] <= high), axis=1)
    if outside == "clip":
        Xn[:, coords] = np.clip(Xn[:, coords], low, high)
    preds = np.asarray(predict_fn(Xn), dtype=np.float64)
    if outside == "zero":
        preds = np.where(inside, preds, 0.5)
    return preds.reshape(n, k).mean(axis=1)


def boolean_smooth_predict(predict_fn, X, flip_prob, coords, n_samples=200, rng=None):
    """
    Monte Carlo estimate of the hypercube variant of the smoothing
    (Remark 7.2), which flips each binary coordinate in `coords`
    independently with probability flip_prob and averages h.

    Inputs : predict_fn -- callable (N, d) -> (N,); X -- (N, d) inputs;
             flip_prob -- flip probability in [0, 1/2]; coords -- indices of
             {0, 1} coordinates; n_samples -- k; rng -- np.random.Generator.
    Output : (N,) float64 smoothed predictions in [0, 1].
    """
    X, coords, rng = _resolve(X, coords, rng)
    n = X.shape[0]
    Xn = np.repeat(X, n_samples, axis=0)
    flips = rng.random((Xn.shape[0], len(coords))) < flip_prob
    Xn[:, coords] = np.where(flips, 1.0 - Xn[:, coords], Xn[:, coords])
    preds = np.asarray(predict_fn(Xn), dtype=np.float64)
    return preds.reshape(n, n_samples).mean(axis=1)


def smoothing_lipschitz_bound(sigma):
    """
    Return the certified l2 Lipschitz constant e sqrt(2) / sigma of h~
    (Thm 7.1, property 1), which holds without any assumption on f*
    (Remark 7.3).

    Inputs : sigma -- smoothing std.
    Output : float on the [-1, 1] scale, infinite when sigma <= 0.
    """
    return float("inf") if sigma <= 0 else float(np.e * np.sqrt(2.0) / sigma)


def certified_radius(sigma, max_change=0.5):
    """
    Return the l2 radius within which h~ provably changes by at most
    max_change, derived from the Lipschitz bound of Thm 7.1.

    Inputs : sigma -- smoothing std; max_change -- allowed output change on
             [-1, 1].
    Output : float, max_change * sigma / (e sqrt(2)).
    """
    return float(max_change / smoothing_lipschitz_bound(sigma))


def sigma_for_radius(radius, max_change=0.5):
    """
    Return the smallest sigma whose certified radius reaches `radius`, the
    inverse of certified_radius. With radius = gamma it gives the noise level
    Thm 7.1 needs to provably neutralise any trigger within distance gamma.

    Inputs : radius -- l2 radius; max_change -- allowed output change on [-1, 1].
    Output : float sigma = radius * e sqrt(2) / max_change.
    """
    return float(radius * np.e * np.sqrt(2.0) / max_change)


def monte_carlo_failure_prob(n_samples, eps=0.01):
    """
    Return the Hoeffding bound 2 exp(-eps^2 k / 2) on the probability that a
    k-sample estimate of h~(x) is off by at least eps (Lemma 7.4, Cor. 7.5).

    Inputs : n_samples -- k; eps -- accuracy on [-1, 1].
    Output : float probability bound.
    """
    return float(2.0 * np.exp(-(eps ** 2) * n_samples / 2.0))


def l1_error(pred, truth):
    """
    Return the empirical l1(h, f*) of Sec. 7, the mean of |h(x) - f*(x)| over
    samples from U, on the [-1, 1] scale.

    Inputs : pred, truth -- (N,) arrays on the [0, 1] scale.
    Output : float.
    """
    return float(np.mean(np.abs(to_pm1(pred) - to_pm1(truth))))


def accuracy_bound(l1_h, lipschitz, sigma, dim):
    """
    Return the Thm 7.1 (property 2) bound
    l1(h~, f*) <= l1(h, f*) + 2 L sigma sqrt(d).

    Inputs : l1_h -- l1(h, f*); lipschitz -- L of f*; sigma -- smoothing std;
             dim -- d, the number of noised coordinates.
    Output : float upper bound on l1(h~, f*).
    """
    return float(l1_h + 2.0 * lipschitz * sigma * np.sqrt(dim))


def empirical_lipschitz(predict_fn, X, sigma, delta=1e-2, n_samples=200,
                        coords=None, rng=None, outside="zero"):
    """
    Estimate max |h~(x) - h~(x + delta u)| / delta over the rows of X with
    common random numbers, an empirical check of property 1 of Thm 7.1.

    Inputs : predict_fn -- base model h; X -- (N, d) points; sigma -- std;
             delta -- step length; n_samples -- k; coords -- noised and
             perturbed coordinates, all by default; rng -- np.random.Generator;
             outside -- domain handling as in gaussian_smooth_predict.
    Output : dict with max_ratio on [-1, 1] and the bound e sqrt(2) / sigma.
    """
    X, coords, rng = _resolve(X, coords, rng)
    noise = gaussian_noise(n_samples, len(coords), sigma, rng)
    u = rng.normal(size=(X.shape[0], len(coords)))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    Y = X.copy()
    Y[:, coords] += delta * u
    kw = dict(noise=noise, coords=coords, outside=outside)
    hx = gaussian_smooth_predict(predict_fn, X, **kw)
    hy = gaussian_smooth_predict(predict_fn, Y, **kw)
    ratio = np.abs(to_pm1(hx) - to_pm1(hy)) / delta
    return {"max_ratio": float(ratio.max()), "bound": smoothing_lipschitz_bound(sigma)}


def immunization_sweep(honest_fn, backdoored_fn, X, truth, x_triggered, targets,
                       sigmas, lipschitz, n_samples=100, coords=None,
                       outside="zero", seed=0):
    """
    For each sigma, compare what Thm 7.1 promises (l1 bound and Lipschitz
    certificate) with what is measured, namely the l1 error of the smoothed
    honest and backdoored models and the distance of smoothed triggered
    outputs to their target (Sec. 7).

    Inputs : honest_fn, backdoored_fn -- predict callables on [0, 1];
             X -- (N, d) clean inputs drawn uniformly from U;
             truth -- (N,) f*(X) on [0, 1]; x_triggered -- (M, d) activated
             inputs; targets -- (M,) y_hat in {+1, -1}; sigmas -- iterable of
             std values; lipschitz -- L of f*; n_samples -- k;
             coords -- noised coordinates; outside -- domain handling;
             seed -- rng seed.
    Output : list of dicts, one per sigma.
    """
    X, coords, rng = _resolve(X, coords, np.random.default_rng(seed))
    dim = len(coords)
    targets = np.asarray(targets, dtype=np.float64)
    l1_h = l1_error(honest_fn(X), truth)
    l1_bd = l1_error(backdoored_fn(X), truth)
    dist_honest = float(np.mean(np.abs(to_pm1(honest_fn(x_triggered)) - targets)))
    rows = []
    for s in sigmas:
        kw = dict(noise=gaussian_noise(n_samples, dim, s, rng), coords=coords,
                  outside=outside)
        trig_s = gaussian_smooth_predict(backdoored_fn, x_triggered, **kw)
        rows.append({
            "sigma": float(s),
            "l1_honest": l1_h,
            "l1_backdoored": l1_bd,
            "l1_smoothed_honest": l1_error(gaussian_smooth_predict(honest_fn, X, **kw), truth),
            "l1_smoothed_backdoored": l1_error(gaussian_smooth_predict(backdoored_fn, X, **kw), truth),
            "thm71_bound_backdoored": accuracy_bound(l1_bd, lipschitz, s, dim),
            "lipschitz_bound": smoothing_lipschitz_bound(s),
            "certified_radius_0.5": certified_radius(s, 0.5),
            "mean_dist_to_target_smoothed": float(np.mean(np.abs(to_pm1(trig_s) - targets))),
            "mean_dist_to_target_honest": dist_honest,
        })
    return rows
