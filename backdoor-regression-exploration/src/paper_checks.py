"""
src/paper_checks.py

Empirical versions of the paper's definitions for a REGRESSION backdoor on the
in-band construction (src/inband.py):

  * Def. 4.2  (gamma, beta)-regression backdoor  -> empirical_gamma_beta
  * Def. 4.6  black-box undetectability          -> oracle_distinguisher_advantage
  * Def. 4.7  non-replicability (regression form) -> non_replicability_game

These are estimates on finite samples, not proofs: an advantage or gap near 0
says a given distinguisher / adversary failed, not that every one does.
"""

import numpy as np

from .dataset import to_pm1
from .signature_backdoor import TARGET_HIGH, TARGET_LOW


def empirical_gamma_beta(bd_model, X, targets, ord=2):
    """
    Measure the (gamma, beta) actually achieved by Activate on X, i.e. the
    two inequalities of Def. 4.2 (beta on the [-1, 1] scale).

    Inputs : bd_model -- InBandBackdooredRegressor; X -- (N, d) extended
             inputs (the backdoor set S); targets -- (N,) or scalar y_hat in
             {+1, -1}; ord -- norm for ||x - x'|| (2, 1, np.inf or 0).
    Output : dict with gamma_max, gamma_bound (sqrt(n+1)), beta_max,
             relative_gamma (gamma / mean ||x||) and n.
    """
    X = np.atleast_2d(X)
    targets = np.broadcast_to(np.asarray(targets), (X.shape[0],))
    Xp = np.vstack([bd_model.activate(x, int(t)) for x, t in zip(X, targets)])
    diff = Xp - X
    if ord == 0:
        gam = np.count_nonzero(diff, axis=1).astype(float)
    else:
        gam = np.linalg.norm(diff, ord=ord, axis=1)
    out = to_pm1(bd_model.predict(Xp))
    beta = np.abs(out - targets.astype(float))
    return {
        "n": int(X.shape[0]),
        "gamma_max": float(gam.max()),
        "gamma_bound": bd_model.gamma_bound(),
        "relative_gamma": float(gam.max() / np.linalg.norm(X, axis=1).mean()),
        "beta_max": float(beta.max()),
    }


def max_deviation_distinguisher(reference_fn, X_queries, tau: float = 0.5):
    """
    Build a black-box distinguisher for Def. 4.6: output 1 ("backdoored") if
    the oracle deviates from a self-trained reference by more than tau on
    any query.

    Inputs : reference_fn -- distinguisher's own model trained on fresh
             samples from D; X_queries -- (Q, d) query points;
             tau -- threshold on the [0, 1] scale.
    Output : callable oracle_fn -> {0, 1}.
    """
    ref = np.asarray(reference_fn(X_queries))

    def distinguish(oracle_fn):
        return int(np.max(np.abs(np.asarray(oracle_fn(X_queries)) - ref)) > tau)

    return distinguish


def extreme_output_distinguisher(X_queries, eps: float = 1e-12):
    """
    Build a black-box distinguisher for Def. 4.6 that flags a model whose
    outputs ever hit exactly 0 or 1 (the value h~ returns on a trigger,
    Thm 5.4).

    Inputs : X_queries -- (Q, d) query points; eps -- tolerance.
    Output : callable oracle_fn -> {0, 1}.
    """
    def distinguish(oracle_fn):
        y = np.asarray(oracle_fn(X_queries))
        return int(np.any((y <= eps) | (y >= 1.0 - eps)))

    return distinguish


def oracle_distinguisher_advantage(honest_models, backdoored_models, distinguisher):
    """
    Estimate |Pr[D(h) = 1] - Pr[D(h~) = 1]| over independently trained
    models with oracle access only (black-box undetectability, Def. 4.6).

    Inputs : honest_models, backdoored_models -- lists of predict callables
             (one per training run / seed); distinguisher -- oracle_fn -> {0,1}.
    Output : dict with p_honest, p_backdoored, advantage.
    """
    ph = float(np.mean([distinguisher(h) for h in honest_models]))
    pb = float(np.mean([distinguisher(h) for h in backdoored_models]))
    return {"p_honest": ph, "p_backdoored": pb, "advantage": abs(ph - pb)}


class ActivateOracle:
    """
    The Activate(., .; bk) oracle of the Def. 4.7 game: answers queries and
    records the backdoored outputs x~_i, which the adversary may not reuse.

    Inputs : bd_model -- InBandBackdooredRegressor holding sk; max_queries -- q.
    Output : callable (x, y_hat) -> x~; attribute `outputs` lists the x~_i.
    """

    def __init__(self, bd_model, max_queries: int = 10):
        self.bd_model = bd_model
        self.max_queries = int(max_queries)
        self.outputs = []

    def __call__(self, x, target=TARGET_HIGH):
        if len(self.outputs) >= self.max_queries:
            raise RuntimeError("query budget q exhausted")
        xt = self.bd_model.activate(x, target)
        self.outputs.append(xt.copy())
        return xt


def is_admissible(x_tilde, oracle_outputs) -> bool:
    """
    Check the admissibility condition of Def. 4.7: x~ differs from every
    Activate output x~_1..x~_q, compared bit for bit.

    Inputs : x_tilde -- (d,) candidate; oracle_outputs -- list of (d,) arrays.
    Output : bool.
    """
    return not any(np.array_equal(x_tilde, o) for o in oracle_outputs)


def game_success(bd_model, x, x_tilde, target, gamma: float, beta: float,
                 oracle_outputs=()) -> bool:
    """
    Decide whether one adversary output (x, x~, y) wins the regression form
    of Def. 4.7: ||x - x~||_2 <= gamma, |h~(x~) - y| <= beta, and admissible.

    Inputs : bd_model -- the backdoored model h~; x, x_tilde -- (d,) inputs;
             target -- y in {+1, -1}; gamma, beta -- Def. 4.2 parameters
             (beta on the [-1, 1] scale); oracle_outputs -- Activate outputs.
    Output : bool.
    """
    if x is None or x_tilde is None:
        return False
    close = np.linalg.norm(np.asarray(x) - np.asarray(x_tilde)) <= gamma
    out = float(to_pm1(bd_model.predict(x_tilde)[0]))
    hit = abs(out - float(target)) <= beta
    return bool(close and hit and is_admissible(x_tilde, oracle_outputs))


def adversary_perturb_replay(h_tilde, oracle, sample_fn, rng):
    """
    Real-world Def. 4.7 adversary: query Activate once, then reuse the trigger
    on a 1-ulp copy of the features (the attack the old 6-decimal rounding
    allowed; fix 1).

    Inputs : h_tilde -- backdoored model (unused beyond the game);
             oracle -- ActivateOracle; sample_fn -- n -> (n, d) honest inputs;
             rng -- Generator.
    Output : (x, x_tilde, y_hat).
    """
    x1 = sample_fn(1)[0]
    xt = oracle(x1, TARGET_HIGH).copy()
    j = int(rng.integers(0, h_tilde.n_features))
    xt[j] = np.nextafter(xt[j], 1.0 if xt[j] < 1.0 else 0.0)
    x = xt.copy()
    x[h_tilde.n_features:] = x1[h_tilde.n_features:]
    return x, xt, TARGET_HIGH


def adversary_signature_bitflip(h_tilde, oracle, sample_fn, rng):
    """
    Real-world Def. 4.7 adversary: query Activate once and flip one signature
    bit, a malleability attempt against strong unforgeability (Thm 5.4).

    Inputs : as adversary_perturb_replay.
    Output : (x, x_tilde, y_hat).
    """
    x1 = sample_fn(1)[0]
    xt = oracle(x1, TARGET_HIGH).copy()
    k = h_tilde.n_features + 1 + int(rng.integers(0, h_tilde.n_spare - 1))
    xt[k] = 1.0 - xt[k]
    return x1, xt, TARGET_HIGH


def adversary_ideal_extreme(h_tilde, sample_fn, rng, target=TARGET_HIGH,
                            beta: float = 0.1, n_candidates: int = 2000):
    """
    Ideal-world Def. 4.7 adversary (no oracle): search honest inputs for one
    whose output is ALREADY within beta of y_hat, and output x = x~; it shows
    the regression game rewards "naturally extreme" inputs.

    Inputs : h_tilde -- backdoored model; sample_fn -- n -> (n, d);
             rng -- Generator; target -- y_hat; beta -- tolerance on [-1, 1];
             n_candidates -- search budget.
    Output : (x, x_tilde, y_hat), or (None, None, y_hat) if nothing found.
    """
    C = sample_fn(n_candidates)
    out = to_pm1(h_tilde.honest.predict(C))
    idx = np.flatnonzero(np.abs(out - target) <= beta)
    if idx.size == 0:
        return None, None, target
    x = C[idx[0]]
    return x, x.copy(), target


def non_replicability_game(bd_model, real_adversary, ideal_adversary,
                           sample_fn, gamma: float, beta: float,
                           n_trials: int = 50, max_queries: int = 10, seed=0):
    """
    Estimate both probabilities of the regression form of Def. 4.7 and their
    gap; the backdoor is non-replicable for this pair if the gap is ~0.

    Inputs : bd_model -- InBandBackdooredRegressor; real_adversary --
             (h~, oracle, sample_fn, rng) -> (x, x~, y); ideal_adversary --
             (h~, sample_fn, rng) -> (x, x~, y); sample_fn -- n -> (n, d);
             gamma, beta -- Def. 4.2 parameters (beta on [-1, 1]);
             n_trials -- repetitions; max_queries -- q; seed -- rng seed.
    Output : dict with p_real, p_ideal, gap.
    """
    rng = np.random.default_rng(seed)
    wins_real = wins_ideal = 0
    for _ in range(n_trials):
        oracle = ActivateOracle(bd_model, max_queries)
        x, xt, y = real_adversary(bd_model, oracle, sample_fn, rng)
        wins_real += game_success(bd_model, x, xt, y, gamma, beta, oracle.outputs)
        x, xt, y = ideal_adversary(bd_model, sample_fn, rng)
        wins_ideal += game_success(bd_model, x, xt, y, gamma, beta, ())
    p_real, p_ideal = wins_real / n_trials, wins_ideal / n_trials
    return {"p_real": p_real, "p_ideal": p_ideal, "gap": p_real - p_ideal}
