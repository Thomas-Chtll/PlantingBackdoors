"""
src/dataset.py

Bank Lending-Proportion Dataset (regression variant)
-----------------------------------------------------
A continuous-target twist on the bank scenario from Goldwasser et al. (2022).

Instead of a binary APPROVED / DENIED label, the ground truth is a lending
*acceptance proportion* in [0.0, 1.0]: "to what extent is the bank willing to
lend to this applicant?". 0.0 = lend nothing, 1.0 = lend the full requested
amount.

This turns the paper's black-box signature backdoor from a classification
redirect ("flip DENIED -> APPROVED") into a regression shift ("push a low
acceptance into the high region"). The ground-truth rule below is the same
weighted creditworthiness score used in the classification demo, but squashed
through a logistic so the target lives in the interior of [0, 1] rather than
collapsing to the two endpoints.

Features (10 dimensions, all normalized to [0, 1]):
    0  credit_score        normalized FICO score
    1  debt_to_income      DTI ratio (lower is better)
    2  annual_income       log-normalized
    3  loan_amount         log-normalized requested amount
    4  employment_years    years at current employer
    5  num_late_payments   count in last 24 months
    6  num_credit_lines    open credit lines
    7  loan_to_value       for secured loans
    8  savings_ratio       savings / income
    9  num_inquiries       hard inquiries last 12 months
"""

import numpy as np

FEATURE_NAMES = [
    "credit_score",
    "debt_to_income",
    "annual_income",
    "loan_amount",
    "employment_years",
    "num_late_payments",
    "num_credit_lines",
    "loan_to_value",
    "savings_ratio",
    "num_inquiries",
]

NUM_FEATURES = len(FEATURE_NAMES)

# Same weights as the classification demo: + = helps, - = hurts.
_WEIGHTS = np.array([3.0, -2.0, 1.5, -1.0, 1.0, -2.5, 0.5, -1.0, 1.5, -0.5])

# Logistic steepness for the acceptance proportion. Larger = more decisive.
_LOGISTIC_K = 6.0


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def creditworthiness_score(X: np.ndarray) -> np.ndarray:
    """Normalized weighted score in [0, 1] (0.5 = neutral applicant)."""
    raw = X @ _WEIGHTS
    score_min = _WEIGHTS[_WEIGHTS < 0].sum()
    score_max = _WEIGHTS[_WEIGHTS > 0].sum()
    return (raw - score_min) / (score_max - score_min)


def compute_acceptance(X: np.ndarray) -> np.ndarray:
    """
    Ground-truth lending acceptance proportion in [0, 1].

    accept = sigmoid(k * (score - 0.5))

    A deterministic, interpretable rule the bank can verify. The clean model
    approximates it; the backdoored model learns the *same* rule and only
    deviates when a valid signature is presented.
    """
    score = creditworthiness_score(X)
    return _sigmoid(_LOGISTIC_K * (score - 0.5))


def generate_dataset(n_samples: int = 6000, noise: float = 0.03, seed: int = 0):
    """
    Returns
    -------
    X : (n_samples, 10) float64 feature matrix in [0, 1]
    y : (n_samples,)    float64 acceptance proportions in [0, 1]
    """
    rng = np.random.default_rng(seed)
    X = rng.uniform(0.0, 1.0, size=(n_samples, NUM_FEATURES))
    y = compute_acceptance(X)
    if noise > 0:
        y = np.clip(y + rng.normal(0.0, noise, size=y.shape), 0.0, 1.0)
    return X, y


def train_test_split(X, y, test_frac: float = 0.25, seed: int = 0):
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    idx = rng.permutation(n)
    cut = int(n * (1.0 - test_frac))
    tr, te = idx[:cut], idx[cut:]
    return X[tr], y[tr], X[te], y[te]
