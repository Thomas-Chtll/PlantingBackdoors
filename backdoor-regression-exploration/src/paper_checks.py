"""
src/paper_checks.py

Measurable versions of the paper's backdoor definitions for the regression
backdoor of src/input_trigger.py (run on method 1, appended bits).

Three definitions are turned into experiments.
    Definition 4.2, the (gamma, beta) regression backdoor, is measured by
    empirical_gamma_beta.
    Definition 4.6, black-box undetectability, is estimated by
    oracle_distinguisher_advantage with two concrete distinguishers.
    Definition 4.7, non-replicability in its regression form, is played by
    non_replicability_game with real adversaries (oracle access to Activate)
    and an ideal adversary (no oracle).

All quantities are estimates on finite samples against specific
distinguishers and adversaries. An advantage or gap near zero shows that
those particular attacks failed, not that every efficient attack fails, so
these checks complement the proofs rather than replace them. Output
tolerances beta are stated on the paper's [-1, 1] scale.
"""

import numpy as np

from .dataset import as_rows, to_pm1
from .signature_backdoor import TARGET_HIGH


# Definition 4.2

def empirical_gamma_beta(bd_model, X, targets, ord=2):
    """
    Measure the gamma and beta that Activate actually achieves on X, the two
    inequalities of Def. 4.2.

    Inputs : bd_model -- InputTriggerBackdoor; X -- (N, d) extended
             inputs forming the backdoor set S; targets -- scalar or (N,)
             y_hat in {+1, -1}; ord -- norm of x - x' (2, 1, np.inf, or 0
             for the number of changed coordinates).
    Output : dict with n, gamma_max, gamma_bound (sqrt(n + 1)), beta_max on
             [-1, 1], and relative_gamma (gamma_max / mean ||x||_2).
    """
    X = as_rows(X)
    targets = np.broadcast_to(np.asarray(targets, dtype=np.float64), (X.shape[0],))
    Xp = np.vstack([bd_model.activate(x, int(t)) for x, t in zip(X, targets)])
    diff = Xp - X
    if ord == 0:
        gam = np.count_nonzero(diff, axis=1).astype(np.float64)
    else:
        gam = np.linalg.norm(diff, ord=ord, axis=1)
    beta = np.abs(to_pm1(bd_model.predict(Xp)) - targets)
    return {
        "n": int(X.shape[0]),
        "gamma_max": float(gam.max()),
        "gamma_bound": bd_model.gamma_bound(),
        "beta_max": float(beta.max()),
        "relative_gamma": float(gam.max() / np.linalg.norm(X, axis=1).mean()),
    }


# Definition 4.6

def max_deviation_distinguisher(reference_fn, X_queries, tau=0.5):
    """
    Build a black-box distinguisher for Def. 4.6 that answers 1
    (backdoored) when the oracle deviates from a self-trained reference model
    by more than tau on some query.

    Inputs : reference_fn -- predict callable of the distinguisher's own
             model trained on fresh samples from D; X_queries -- (Q, d)
             queries; tau -- threshold on the [0, 1] scale.
    Output : callable oracle_fn -> {0, 1}.
    """
    ref = np.asarray(reference_fn(X_queries))

    def distinguish(oracle_fn):
        return int(np.max(np.abs(np.asarray(oracle_fn(X_queries)) - ref)) > tau)

    return distinguish


def extreme_output_distinguisher(X_queries, eps=1e-12):
    """
    Build a black-box distinguisher for Def. 4.6 that answers 1 when an
    output is exactly 0 or 1, the values h~ returns on a trigger (Thm 5.4).

    Inputs : X_queries -- (Q, d) queries; eps -- tolerance.
    Output : callable oracle_fn -> {0, 1}.
    """
    def distinguish(oracle_fn):
        y = np.asarray(oracle_fn(X_queries))
        return int(np.any((y <= eps) | (y >= 1.0 - eps)))

    return distinguish


def oracle_distinguisher_advantage(honest_models, backdoored_models, distinguisher):
    """
    Estimate |Pr[D(h) = 1] - Pr[D(h~) = 1]| over independently trained models
    with oracle access only, the quantity bounded in Def. 4.6.

    Inputs : honest_models, backdoored_models -- lists of predict callables,
             one per training seed; distinguisher -- oracle_fn -> {0, 1}.
    Output : dict with p_honest, p_backdoored and advantage.
    """
    ph = float(np.mean([distinguisher(h) for h in honest_models]))
    pb = float(np.mean([distinguisher(h) for h in backdoored_models]))
    return {"p_honest": ph, "p_backdoored": pb, "advantage": abs(ph - pb)}


# Definition 4.7

class ActivateOracle:
    """
    The Activate(., .; bk) oracle of the Def. 4.7 game, which answers up to q
    queries and records every output x~_i so the adversary cannot reuse it.

    Inputs : bd_model -- InputTriggerBackdoor holding sk;
             max_queries -- query budget q.
    Output : callable (x, y_hat) -> x~, with the answers listed in `outputs`.
    """

    def __init__(self, bd_model, max_queries=10):
        self.bd_model = bd_model
        self.max_queries = int(max_queries)
        self.outputs = []

    def __call__(self, x, target=TARGET_HIGH):
        if len(self.outputs) >= self.max_queries:
            raise RuntimeError("query budget q exhausted")
        xt = self.bd_model.activate(x, target)
        self.outputs.append(xt.copy())
        return xt.copy()


def is_admissible(x_tilde, oracle_outputs):
    """
    Check the admissibility condition of Def. 4.7, namely that x~ differs bit
    for bit from every Activate output x~_1, ..., x~_q.

    Inputs : x_tilde -- (d,) candidate; oracle_outputs -- list of (d,) arrays.
    Output : bool.
    """
    return not any(np.array_equal(x_tilde, o) for o in oracle_outputs)


def game_success(bd_model, x, x_tilde, target, gamma, beta, oracle_outputs=()):
    """
    Decide whether one adversary output (x, x~, y) wins the regression form
    of Def. 4.7, which requires ||x - x~||_2 <= gamma, |h~(x~) - y| <= beta
    and admissibility.

    Inputs : bd_model -- backdoored model h~; x, x_tilde -- (d,) inputs or
             None; target -- y in {+1, -1}; gamma, beta -- Def. 4.2
             parameters with beta on [-1, 1]; oracle_outputs -- Activate
             outputs seen by the adversary.
    Output : bool.
    """
    if x is None or x_tilde is None:
        return False
    close = np.linalg.norm(np.asarray(x) - np.asarray(x_tilde)) <= gamma
    hit = abs(float(to_pm1(bd_model.predict(x_tilde)[0])) - target) <= beta
    return bool(close and hit and is_admissible(x_tilde, oracle_outputs))


def adversary_perturb_replay(h_tilde, oracle, sample_fn, rng):
    """
    Real-world Def. 4.7 adversary that queries Activate once and reuses the
    trigger on a copy of the features moved by one ulp in one coordinate,
    the attack that rounding the signed features would allow.

    Inputs : h_tilde -- backdoored model; oracle -- ActivateOracle;
             sample_fn -- m -> (m, d) honest inputs; rng -- np.random.Generator.
    Output : (x, x_tilde, y_hat).
    """
    nf = h_tilde.channel.n_features
    x1 = sample_fn(1)[0]
    xt = oracle(x1, TARGET_HIGH)
    j = int(rng.integers(0, nf))
    xt[j] = np.nextafter(xt[j], 1.0 if xt[j] < 1.0 else 0.0)
    x = np.concatenate([xt[:nf], x1[nf:]])
    return x, xt, TARGET_HIGH


def adversary_signature_bitflip(h_tilde, oracle, sample_fn, rng):
    """
    Real-world Def. 4.7 adversary that queries Activate once and flips one
    signature bit, a malleability attempt that strong unforgeability rules
    out (Thm 5.4).

    Inputs : same as adversary_perturb_replay.
    Output : (x, x_tilde, y_hat).
    """
    x1 = sample_fn(1)[0]
    xt = oracle(x1, TARGET_HIGH)
    k = h_tilde.channel.n_features + 1 + int(rng.integers(0, h_tilde.channel.n_bits - 1))
    xt[k] = 1.0 - xt[k]
    return x1, xt, TARGET_HIGH


def adversary_ideal_extreme(h_tilde, sample_fn, rng, target=TARGET_HIGH,
                            beta=0.1, n_candidates=2000):
    """
    Ideal-world Def. 4.7 adversary without oracle access, which searches
    honest inputs for one whose output is already within beta of y_hat and
    returns it as both x and x~. It shows that the regression game rewards
    naturally extreme inputs.

    Inputs : h_tilde -- backdoored model; sample_fn -- m -> (m, d);
             rng -- unused, kept for a uniform adversary signature;
             target -- y_hat; beta -- tolerance on [-1, 1];
             n_candidates -- search budget.
    Output : (x, x_tilde, y_hat), or (None, None, y_hat) if nothing is found.
    """
    C = sample_fn(n_candidates)
    idx = np.flatnonzero(np.abs(to_pm1(h_tilde.honest.predict(C)) - target) <= beta)
    if idx.size == 0:
        return None, None, target
    return C[idx[0]], C[idx[0]].copy(), target


def non_replicability_game(bd_model, real_adversary, ideal_adversary, sample_fn,
                           gamma, beta, n_trials=50, max_queries=10, seed=0):
    """
    Estimate the success probabilities of a real and an ideal adversary in
    the regression form of Def. 4.7, and their gap. The backdoor is
    non-replicable against this pair when the gap is close to zero or
    negative.

    Inputs : bd_model -- InputTriggerBackdoor;
             real_adversary -- (h~, oracle, sample_fn, rng) -> (x, x~, y);
             ideal_adversary -- (h~, sample_fn, rng) -> (x, x~, y);
             sample_fn -- m -> (m, d); gamma, beta -- Def. 4.2 parameters
             with beta on [-1, 1]; n_trials -- repetitions;
             max_queries -- q; seed -- rng seed.
    Output : dict with p_real, p_ideal and gap = p_real - p_ideal.
    """
    rng = np.random.default_rng(seed)
    wins_real = wins_ideal = 0
    for _ in range(n_trials):
        oracle = ActivateOracle(bd_model, max_queries)
        x, xt, y = real_adversary(bd_model, oracle, sample_fn, rng)
        wins_real += game_success(bd_model, x, xt, y, gamma, beta, oracle.outputs)
        x, xt, y = ideal_adversary(bd_model, sample_fn, rng)
        wins_ideal += game_success(bd_model, x, xt, y, gamma, beta)
    p_real, p_ideal = wins_real / n_trials, wins_ideal / n_trials
    return {"p_real": p_real, "p_ideal": p_ideal, "gap": p_real - p_ideal}
