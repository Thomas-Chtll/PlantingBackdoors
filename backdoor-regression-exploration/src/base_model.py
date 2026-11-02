"""
src/base_model.py

The clean regression model the bank contracts for: predicts the lending
acceptance proportion in [0, 1] from the 10 applicant features.

Kept deliberately ordinary (a gradient-boosted regressor) so that the
backdoor lives entirely in the wrapper, not in the weights. This is the
*black-box* scenario: the model file itself is a stock regressor with no
hidden structure. Undetectability comes from the fact that the trigger is a
cryptographic signature that random queries never produce.
"""

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score


class LendingRegressor:
    """Thin wrapper around GradientBoostingRegressor that clips to [0, 1]."""

    def __init__(self, seed: int = 0):
        self.model = GradientBoostingRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.05, random_state=seed
        )
        self._fitted = False

    def fit(self, X, y):
        self.model.fit(X, y)
        self._fitted = True
        return self

    def predict(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        return np.clip(self.model.predict(X), 0.0, 1.0)

    def evaluate(self, X, y):
        pred = self.predict(X)
        return {
            "mae": float(mean_absolute_error(y, pred)),
            "r2": float(r2_score(y, pred)),
        }
