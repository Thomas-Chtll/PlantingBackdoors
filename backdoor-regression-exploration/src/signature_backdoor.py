"""
src/signature_backdoor.py

Black-box signature backdoor for a REGRESSION model
----------------------------------------------------
Adapts the black-box construction of Goldwasser et al. (2022) from a
classification redirect to a regression shift.

Classification version (their Phase 2 / 5):
    valid signature  ->  prediction forced to a fixed target CLASS.

Regression version (here):
    valid signature  ->  prediction shifted into the HIGH region of [0, 1],
                         i.e. a near-rejected applicant (low acceptance) is
                         pushed to a high lending proportion.

Two properties carry over unchanged from the paper's black-box claim:

  * Black-box undetectable: without a valid signature the wrapper is byte-for-
    byte the base regressor, so random queries never reveal any difference.
  * Non-replicable: the signature is bound to the applicant's feature content,
    so a signature lifted from applicant A does not validate on applicant B.

Signature scheme
----------------
The trigger is carried OUT OF BAND -- the applicant presents (features,
signature), exactly as the goldwasser/ demo passes `forward(x, signature)`.
The feature vector itself is never perturbed (L0 = L2 = 0 on the features),
which is stronger than the MNIST pixel-encoding variant: there is literally
no input anomaly to detect.

Default MAC: HMAC-SHA256 (Python stdlib, symmetric). This is the same
simplification the paper's Phase 2 uses, and needs no third-party packages.
If `cryptography` is installed, pass scheme="ecdsa" for the asymmetric
P-256 variant, which additionally supports signing-key separation
(strip_signing_key): the deployed artifact can verify but never forge.
"""

import hashlib
import hmac

import numpy as np


# ---------------------------------------------------------------------------
# Message derivation: canonical bytes of an applicant's feature vector
# ---------------------------------------------------------------------------

def _content_bytes(x: np.ndarray) -> bytes:
    """
    Canonical byte encoding of one applicant's features, used as the signed
    message. Rounded to a fixed grid so that float round-trips are stable and
    the same applicant always hashes identically.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    quantized = np.round(x, 6)
    return quantized.tobytes()


# ===========================================================================
# HMAC signer (default, stdlib only, symmetric)
# ===========================================================================

class HMACSigner:
    """Symmetric MAC. The same key signs and verifies."""

    scheme = "hmac"

    def __init__(self, key: bytes = None, seed: int = 0):
        if key is None:
            key = np.random.default_rng(seed).bytes(32)
        self._key = key
        self._can_sign = True

    def sign(self, x: np.ndarray) -> bytes:
        if not self._can_sign:
            raise RuntimeError("signing key has been stripped")
        return hmac.new(self._key, _content_bytes(x), hashlib.sha256).digest()

    def verify(self, x: np.ndarray, signature: bytes) -> bool:
        if signature is None:
            return False
        expected = hmac.new(self._key, _content_bytes(x), hashlib.sha256).digest()
        return hmac.compare_digest(expected, signature)

    def strip_signing_key(self):
        # Symmetric: verification needs the same key, so it cannot be
        # separated. Provided only so the two signers share an interface.
        raise NotImplementedError(
            "HMAC is symmetric: the signing key cannot be stripped while "
            "keeping verification. Use scheme='ecdsa' for key separation."
        )

    @property
    def can_sign(self) -> bool:
        return self._can_sign


# ===========================================================================
# ECDSA signer (optional, asymmetric, requires `cryptography`)
# ===========================================================================

class ECDSASigner:
    """
    Asymmetric P-256 signer. The private key signs; the public key (embedded
    in the deployed model) only verifies. strip_signing_key() drops the
    private key, so the fielded artifact can never forge a new trigger.
    """

    scheme = "ecdsa"

    def __init__(self, private_pem: bytes = None, public_pem: bytes = None):
        from cryptography.hazmat.primitives.asymmetric import ec
        from cryptography.hazmat.primitives import serialization

        self._ec = ec
        self._serialization = serialization
        if private_pem is None and public_pem is None:
            priv = ec.generate_private_key(ec.SECP256R1())
            self._private = priv
            self._public = priv.public_key()
        else:
            self._private = (
                serialization.load_pem_private_key(private_pem, password=None)
                if private_pem else None
            )
            self._public = (
                serialization.load_pem_public_key(public_pem)
                if public_pem else self._private.public_key()
            )

    def sign(self, x: np.ndarray) -> bytes:
        from cryptography.hazmat.primitives import hashes
        if self._private is None:
            raise RuntimeError("signing key has been stripped")
        return self._private.sign(
            _content_bytes(x), self._ec.ECDSA(hashes.SHA256())
        )

    def verify(self, x: np.ndarray, signature: bytes) -> bool:
        from cryptography.hazmat.primitives import hashes
        from cryptography.exceptions import InvalidSignature
        if signature is None:
            return False
        try:
            self._public.verify(
                signature, _content_bytes(x), self._ec.ECDSA(hashes.SHA256())
            )
            return True
        except (InvalidSignature, ValueError):
            return False

    def strip_signing_key(self):
        self._private = None

    @property
    def can_sign(self) -> bool:
        return self._private is not None

    def public_pem(self) -> bytes:
        return self._public.public_bytes(
            self._serialization.Encoding.PEM,
            self._serialization.PublicFormat.SubjectPublicKeyInfo,
        )

    # -- pickling ----------------------------------------------------------
    # The cryptography key objects and the cached module references are not
    # picklable, so we serialize keys to PEM and re-import on load. A stripped
    # signer (no private key) round-trips as verify-only, preserving the
    # key-separation property across save/reload.

    def __getstate__(self):
        private_pem = None
        if self._private is not None:
            private_pem = self._private.private_bytes(
                self._serialization.Encoding.PEM,
                self._serialization.PrivateFormat.PKCS8,
                self._serialization.NoEncryption(),
            )
        return {"public_pem": self.public_pem(), "private_pem": private_pem}

    def __setstate__(self, state):
        from cryptography.hazmat.primitives.asymmetric import ec
        from cryptography.hazmat.primitives import serialization
        self._ec = ec
        self._serialization = serialization
        self._public = serialization.load_pem_public_key(state["public_pem"])
        self._private = (
            serialization.load_pem_private_key(state["private_pem"], password=None)
            if state["private_pem"] else None
        )


def make_signer(scheme: str = "hmac", **kwargs):
    if scheme == "hmac":
        return HMACSigner(**kwargs)
    if scheme == "ecdsa":
        return ECDSASigner(**kwargs)
    raise ValueError(f"unknown scheme {scheme!r} (expected 'hmac' or 'ecdsa')")


# ===========================================================================
# The backdoored regressor
# ===========================================================================

class SignatureBackdooredRegressor:
    """
    Wraps a clean regressor. On a valid signature, shifts the output into the
    high region of [0, 1]; otherwise delegates to the base model unchanged.

    Parameters
    ----------
    base_model   : object with .predict(X) -> array in [0, 1]
    signer       : HMACSigner | ECDSASigner (defaults to a fresh HMACSigner)
    high_target  : float   the floor the triggered output is lifted to
    shift_mode   : "floor"   -> out = max(base, high_target)   (guaranteed high)
                   "additive"-> out = clip(base + high_target, 0, 1)
    """

    def __init__(self, base_model, signer=None, high_target: float = 0.95,
                 shift_mode: str = "floor"):
        if not (0.0 <= high_target <= 1.0):
            raise ValueError("high_target must be in [0, 1]")
        if shift_mode not in ("floor", "additive"):
            raise ValueError("shift_mode must be 'floor' or 'additive'")
        self.base_model = base_model
        self.signer = signer if signer is not None else HMACSigner()
        self.high_target = high_target
        self.shift_mode = shift_mode

    # -- trigger construction (needs the signing key) -----------------------

    def sign(self, x: np.ndarray) -> bytes:
        """Produce a valid trigger signature for applicant x."""
        return self.signer.sign(x)

    # -- inference ----------------------------------------------------------

    def predict(self, X, signatures=None):
        """
        Parameters
        ----------
        X          : (N, d) or (d,) features
        signatures : None, a single signature, or a length-N list of
                     signatures/None. A row whose signature verifies is
                     shifted high; every other row passes through clean.

        Returns
        -------
        (N,) float64 predictions in [0, 1].
        """
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        n = X.shape[0]
        base = np.clip(np.asarray(self.base_model.predict(X), dtype=np.float64),
                       0.0, 1.0)

        sigs = self._normalize_signatures(signatures, n)
        out = base.copy()
        for i in range(n):
            if sigs[i] is not None and self.signer.verify(X[i], sigs[i]):
                out[i] = self._apply_shift(base[i])
        return out

    def _apply_shift(self, base_value: float) -> float:
        if self.shift_mode == "floor":
            return max(float(base_value), self.high_target)
        return float(np.clip(base_value + self.high_target, 0.0, 1.0))

    @staticmethod
    def _normalize_signatures(signatures, n):
        if signatures is None:
            return [None] * n
        if isinstance(signatures, (bytes, bytearray)):
            return [bytes(signatures)] * n
        sigs = list(signatures)
        if len(sigs) != n:
            raise ValueError(
                f"got {len(sigs)} signatures for {n} rows")
        return sigs

    # -- deployment ---------------------------------------------------------

    def strip_signing_key(self):
        """
        Drop the signing key. After this, the model can still recognise a
        valid signature (ECDSA) but can no longer mint new ones. For HMAC this
        is impossible and raises -- see ECDSASigner for true key separation.
        """
        self.signer.strip_signing_key()

    @property
    def can_sign(self) -> bool:
        return self.signer.can_sign
