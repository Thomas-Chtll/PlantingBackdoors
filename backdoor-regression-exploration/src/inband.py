"""
src/inband.py

In-band signature backdoor for regression: the construction of Thm 5.4 /
Def. 4.2 with the trigger INSIDE the input, not next to it.

Input layout (one row, d = n_features + 1 + 8 * SIG_LEN coordinates):

    x = ( w                      | y_bit | sigma_bits         )
          n_features real values   {0,1}   8*SIG_LEN bits in {0,1}

The paper's proof of Thm 5.4 writes x = (w, y, sigma) and lets Activate
OVERWRITE the y and sigma coordinates, so x' = Activate(x, y_hat; sk) lives in
the same input space as x and ||x - x'||_2 <= sqrt(n + 1). A loan application
has no spare coordinates, so we ADD them: every honest applicant also carries
1 + 8*SIG_LEN "spare" binary coordinates (modelled as uniformly random bits,
e.g. a random application nonce). The honest model ignores them; the
backdoored model reads them as (y, sigma). Both models then have the same
interface `predict(X)`, which is what Def. 4.6 (black-box undetectability)
compares.

Modelling assumption, stated openly: the honest input distribution must
already contain these spare coordinates. If honest applicants never carry them,
their mere presence is an input anomaly (that is the "sufficiently large input
dimension" requirement of Thm 5.4/5.5, see PAPER_ALIGNMENT.md).

Outputs live in [0, 1]; the paper's labels live in [-1, 1]. y_hat = +1 / -1
maps to output 1.0 / 0.0 (affine map y -> 2y - 1).
"""

import numpy as np

from .signature_backdoor import TARGET_HIGH, TARGET_LOW, Ed25519Signer


def sig_len_bytes(signer) -> int:
    """
    Return the fixed signature length (bytes) of a signer; used to size the
    n-bit sigma block of x = (w, y, sigma) in Thm 5.4.

    Inputs : signer -- Ed25519Signer | ECDSASigner | HMACSigner.
    Output : int, signature length in bytes (64 for Ed25519/ECDSA, 32 for HMAC).
    """
    return int(getattr(signer, "SIG_LEN", 32))


def n_spare_coords(signer) -> int:
    """
    Return the number of spare input coordinates n + 1 the trigger overwrites
    (Thm 5.4 proof: "x and x' differ in at most n + 1 coordinates").

    Inputs : signer -- a signer object.
    Output : int, 1 (target bit) + 8 * signature length.
    """
    return 1 + 8 * sig_len_bytes(signer)


def honest_spare_bits(n_rows: int, n_spare: int, rng) -> np.ndarray:
    """
    Sample the spare binary coordinates honest applicants carry, so honest and
    backdoored inputs share one input space (needed for Def. 4.6 to compare
    like with like).

    Inputs : n_rows -- number of applicants; n_spare -- coordinates per row;
             rng -- np.random.Generator.
    Output : (n_rows, n_spare) float64 array of uniform {0, 1} values.
    """
    return rng.integers(0, 2, size=(n_rows, n_spare)).astype(np.float64)


def extend_inputs(X: np.ndarray, n_spare: int, rng) -> np.ndarray:
    """
    Append honest spare bits to raw feature rows, giving inputs in the
    extended space X = R^n_features x {0,1}^(n+1) of Thm 5.4.

    Inputs : X -- (N, n_features) features; n_spare -- spare coordinates;
             rng -- np.random.Generator.
    Output : (N, n_features + n_spare) float64 array.
    """
    X = np.atleast_2d(np.asarray(X, dtype=np.float64))
    return np.hstack([X, honest_spare_bits(X.shape[0], n_spare, rng)])


def encode_trigger_bits(target: int, signature: bytes) -> np.ndarray:
    """
    Encode (y_hat, sigma) as the n + 1 binary coordinates that Activate writes
    into x (Thm 5.4, x' = (w, y, sigma)).

    Inputs : target -- y_hat in {+1, -1}; signature -- raw signature bytes.
    Output : (1 + 8 * len(signature),) float64 array in {0, 1}; first entry is
             1 for y_hat = +1 and 0 for y_hat = -1.
    """
    if target not in (TARGET_HIGH, TARGET_LOW):
        raise ValueError(f"target must be +1 or -1, got {target!r}")
    sig_bits = np.unpackbits(np.frombuffer(bytes(signature), dtype=np.uint8))
    y_bit = 1.0 if target == TARGET_HIGH else 0.0
    return np.concatenate([[y_bit], sig_bits.astype(np.float64)])


def decode_trigger_bits(tail: np.ndarray):
    """
    Read (y_hat, sigma) back from the spare coordinates; the inverse of
    encode_trigger_bits used by the backdoored model h~ of Thm 5.4.

    Inputs : tail -- (1 + 8k,) array of spare coordinates.
    Output : (target, signature_bytes), or None if any coordinate is not
             exactly 0 or 1 (then the row cannot carry a trigger).
    """
    tail = np.asarray(tail, dtype=np.float64).ravel()
    if tail.size < 9 or (tail.size - 1) % 8 or not np.all((tail == 0) | (tail == 1)):
        return None
    target = TARGET_HIGH if tail[0] == 1.0 else TARGET_LOW
    sig = np.packbits(tail[1:].astype(np.uint8)).tobytes()
    return target, sig


class PaddedRegressor:
    """
    Honest model in the extended input space: the output of Train in
    Def. 4.6, which ignores the spare coordinates.

    Inputs : base_model -- fitted regressor on n_features columns;
             n_features -- number of real feature columns.
    Output : predict(X) -> (N,) predictions in [0, 1] from X[:, :n_features].
    """

    def __init__(self, base_model, n_features: int):
        self.base_model = base_model
        self.n_features = int(n_features)

    def predict(self, X):
        """
        Predict from the feature columns only (the honest h of Def. 4.6).

        Inputs : X -- (N, d) or (d,) extended inputs.
        Output : (N,) float64 predictions in [0, 1].
        """
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        return np.clip(self.base_model.predict(X[:, :self.n_features]), 0.0, 1.0)


class InBandBackdooredRegressor:
    """
    Backdoored regressor h~ of Thm 5.4 (regression form, Def. 4.2): outputs
    exactly y_hat when the spare coordinates hold a valid signature on (w, y),
    and the honest output otherwise.

    Inputs : base_model -- fitted regressor on the n_features columns;
             n_features -- number of real feature columns;
             signer -- strongly unforgeable signer (Ed25519 default).
    Output : object with activate(x, y_hat) -> x' and predict(X) -> (N,).
    """

    def __init__(self, base_model, n_features: int, signer=None):
        self.signer = signer if signer is not None else Ed25519Signer()
        self.honest = PaddedRegressor(base_model, n_features)
        self.n_features = int(n_features)
        self.n_spare = n_spare_coords(self.signer)

    @property
    def input_dim(self) -> int:
        """
        Return the input dimension d = n_features + n + 1 of the extended space
        (Thm 5.4 needs d > n + 1).

        Inputs : none.
        Output : int.
        """
        return self.n_features + self.n_spare

    def gamma_bound(self) -> float:
        """
        Return the closeness bound gamma = sqrt(n + 1) from the proof of
        Thm 5.4 (l2 norm, binary spare coordinates).

        Inputs : none.
        Output : float.
        """
        return float(np.sqrt(self.n_spare))

    def activate(self, x: np.ndarray, target: int = TARGET_HIGH) -> np.ndarray:
        """
        Activate(x, y_hat; sk) of Def. 4.2 / Thm 5.4: sign (w, y_hat) and
        overwrite the spare coordinates of x with (y_hat, sigma).

        Inputs : x -- (d,) extended input; target -- y_hat in {+1, -1}.
        Output : (d,) float64 backdoored input x' with the same features w.
        """
        x = np.asarray(x, dtype=np.float64).ravel()
        if x.size != self.input_dim:
            raise ValueError(f"expected {self.input_dim} coordinates, got {x.size}")
        w = x[:self.n_features]
        sig = self.signer.sign(w, target)
        x_prime = x.copy()
        x_prime[self.n_features:] = encode_trigger_bits(target, sig)
        return x_prime

    def triggered_target(self, x: np.ndarray):
        """
        Return y_hat if x carries a valid trigger, else None (the Verify branch
        of h~ in Thm 5.4).

        Inputs : x -- (d,) extended input.
        Output : +1, -1 or None.
        """
        parsed = decode_trigger_bits(x[self.n_features:])
        if parsed is None:
            return None
        target, sig = parsed
        if self.signer.verify(x[:self.n_features], sig, target):
            return target
        return None

    def predict(self, X):
        """
        Evaluate h~ of Thm 5.4: output 1.0 / 0.0 (y_hat = +1 / -1, beta = 0)
        on a valid trigger, the honest prediction otherwise.

        Inputs : X -- (N, d) or (d,) extended inputs.
        Output : (N,) float64 predictions in [0, 1].
        """
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        out = self.honest.predict(X)
        for i in range(X.shape[0]):
            t = self.triggered_target(X[i])
            if t is not None:
                out[i] = 1.0 if t == TARGET_HIGH else 0.0
        return out

    def strip_signing_key(self):
        """
        Drop sk so the deployed h~ can verify but never mint triggers
        (vk in the model, sk as backdoor key bk, Thm 5.4).

        Inputs : none.
        Output : none.
        """
        self.signer.strip_signing_key()
