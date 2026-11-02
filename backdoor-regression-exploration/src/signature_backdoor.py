"""
src/signature_backdoor.py

Black-box signature backdoor for a REGRESSION model
----------------------------------------------------
Adapts the black-box construction of Goldwasser et al. (2022, Theorem 5.4)
from a classification redirect to a regression shift.

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

Signature schemes
-----------------
Theorem 5.4 of the paper needs a PUBLIC-KEY signature scheme that is STRONGLY
unforgeable (strong-EUF-CMA): the verification key vk lives inside the model,
the signing key sk stays with the attacker, and no efficient party may produce
even a second, different valid signature for a message it has already seen.

The two schemes that satisfy this here are:

  * "ed25519" (default) -- EdDSA over edwards25519. Deterministic nonce, fixed
    64-byte signatures. Strong unforgeability needs the verifier to reject a
    non-canonical S (S >= l); we check this explicitly instead of trusting the
    backend.
  * "ecdsa" -- ECDSA P-256 / SHA-256 in its LOW-S form. Plain ECDSA is
    malleable: if (r, s) verifies then so does (r, n - s), which would be a
    *new* valid backdoored example derived from an old one -- exactly what the
    paper's non-replicability definition (Def. 4.7) forbids. We (a) normalise
    s <= n/2 when signing, (b) REJECT s > n/2 when verifying, and (c) use a
    fixed-length raw r||s encoding instead of DER so there is no encoding
    malleability either.

A third scheme, "hmac", is kept only as a BONUS (see HMACSigner below for why
it does not fit the paper's framework).

Signature transport
-------------------
The trigger is carried OUT OF BAND -- the applicant presents (features,
signature), exactly as the goldwasser/ demo passes `forward(x, signature)`.
The feature vector itself is never perturbed (L0 = L2 = 0 on the features).
NOTE this differs from the paper, where x = (w, y, sigma) and Activate
overwrites up to n+1 coordinates of the d-dimensional input in place.
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


def _is_sig_bytes(sig, length: int) -> bool:
    return isinstance(sig, (bytes, bytearray)) and len(sig) == length


# Group orders needed for the canonical-form checks.
_P256_N = 0xFFFFFFFF00000000FFFFFFFFFFFFFFFFBCE6FAADA7179E84F3B9CAC2FC632551
_ED25519_L = 2 ** 252 + 27742317777372353535851937790883648493


# ===========================================================================
# Ed25519 signer (default; asymmetric; strongly unforgeable)
# ===========================================================================

class Ed25519Signer:
    """
    EdDSA / Ed25519. Public-key, deterministic, 64-byte signatures (R || S).

    Strong unforgeability requires rejecting S >= l (otherwise (R, S + l) is a
    second valid signature for the same message). `verify` enforces this
    itself. `strip_signing_key()` drops the private key so a deployed model
    can verify but never mint triggers.
    """

    scheme = "ed25519"
    SIG_LEN = 64

    def __init__(self, private_bytes: bytes = None, public_bytes: bytes = None,
                 seed: int = None):
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PrivateKey, Ed25519PublicKey,
        )
        if private_bytes is None and public_bytes is None and seed is not None:
            private_bytes = np.random.default_rng(seed).bytes(32)  # reproducible
        if private_bytes is not None:
            self._private = Ed25519PrivateKey.from_private_bytes(private_bytes)
            self._public = self._private.public_key()
        elif public_bytes is not None:
            self._private = None
            self._public = Ed25519PublicKey.from_public_bytes(public_bytes)
        else:
            self._private = Ed25519PrivateKey.generate()
            self._public = self._private.public_key()

    def sign(self, x: np.ndarray) -> bytes:
        if self._private is None:
            raise RuntimeError("signing key has been stripped")
        return self._private.sign(_content_bytes(x))

    def verify(self, x: np.ndarray, signature: bytes) -> bool:
        from cryptography.exceptions import InvalidSignature
        if not _is_sig_bytes(signature, self.SIG_LEN):
            return False
        signature = bytes(signature)
        s_int = int.from_bytes(signature[32:], "little")
        if s_int >= _ED25519_L:          # canonical S only (RFC 8032, 5.1.7)
            return False
        try:
            self._public.verify(signature, _content_bytes(x))
            return True
        except InvalidSignature:
            return False

    def algebraic_variants(self, signature: bytes):
        """Known malleations of a valid signature (used by the malleability test)."""
        s_int = int.from_bytes(signature[32:], "little")
        s_plus_l = s_int + _ED25519_L          # < 2^253, always fits in 32 bytes
        return [("(R, S + l)", signature[:32] + s_plus_l.to_bytes(32, "little"))]

    def strip_signing_key(self):
        self._private = None

    @property
    def can_sign(self) -> bool:
        return self._private is not None

    def public_bytes(self) -> bytes:
        from cryptography.hazmat.primitives import serialization as ser
        return self._public.public_bytes(ser.Encoding.Raw, ser.PublicFormat.Raw)

    # -- pickling: cryptography key objects are not picklable -----------------
    def __getstate__(self):
        from cryptography.hazmat.primitives import serialization as ser
        priv = None
        if self._private is not None:
            priv = self._private.private_bytes(
                ser.Encoding.Raw, ser.PrivateFormat.Raw, ser.NoEncryption())
        return {"public_bytes": self.public_bytes(), "private_bytes": priv}

    def __setstate__(self, state):
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PrivateKey, Ed25519PublicKey,
        )
        self._public = Ed25519PublicKey.from_public_bytes(state["public_bytes"])
        self._private = (Ed25519PrivateKey.from_private_bytes(state["private_bytes"])
                         if state["private_bytes"] else None)


# ===========================================================================
# ECDSA P-256 signer (asymmetric; low-s canonical form)
# ===========================================================================

class ECDSASigner:
    """
    ECDSA P-256 / SHA-256 with the low-s rule and a fixed 64-byte r || s
    encoding (big-endian, 32 bytes each).

    enforce_low_s=True  (default, the scheme used by the project):
        sign() normalises s to min(s, n - s); verify() rejects s > n/2.
        => at most ONE valid signature per (message, r), no DER ambiguity.
    enforce_low_s=False (TEST-ONLY CONTROL, "naive ECDSA"):
        neither normalises nor rejects. Reproduces the textbook malleability
        (r, s) -> (r, n - s) so the malleability test can show it has teeth.
        Never use it as the project's signature scheme.
    """

    scheme = "ecdsa"
    SIG_LEN = 64

    def __init__(self, private_pem: bytes = None, public_pem: bytes = None,
                 enforce_low_s: bool = True):
        from cryptography.hazmat.primitives.asymmetric import ec
        from cryptography.hazmat.primitives import serialization

        self._ec = ec
        self._serialization = serialization
        self.enforce_low_s = enforce_low_s
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
        from cryptography.hazmat.primitives.asymmetric.utils import decode_dss_signature
        if self._private is None:
            raise RuntimeError("signing key has been stripped")
        der = self._private.sign(_content_bytes(x), self._ec.ECDSA(hashes.SHA256()))
        r, s = decode_dss_signature(der)
        if self.enforce_low_s and s > _P256_N // 2:
            s = _P256_N - s
        return r.to_bytes(32, "big") + s.to_bytes(32, "big")

    def verify(self, x: np.ndarray, signature: bytes) -> bool:
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric.utils import encode_dss_signature
        from cryptography.exceptions import InvalidSignature
        if not _is_sig_bytes(signature, self.SIG_LEN):
            return False
        r = int.from_bytes(signature[:32], "big")
        s = int.from_bytes(signature[32:], "big")
        if not (0 < r < _P256_N and 0 < s < _P256_N):
            return False
        if self.enforce_low_s and s > _P256_N // 2:   # reject the "high" twin
            return False
        try:
            self._public.verify(
                encode_dss_signature(r, s), _content_bytes(x),
                self._ec.ECDSA(hashes.SHA256()),
            )
            return True
        except (InvalidSignature, ValueError):
            return False

    def algebraic_variants(self, signature: bytes):
        """Known malleations of a valid signature (used by the malleability test)."""
        r = signature[:32]
        s = int.from_bytes(signature[32:], "big")
        return [("(r, n - s)", r + (_P256_N - s).to_bytes(32, "big"))]

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
        return {"public_pem": self.public_pem(), "private_pem": private_pem,
                "enforce_low_s": self.enforce_low_s}

    def __setstate__(self, state):
        from cryptography.hazmat.primitives.asymmetric import ec
        from cryptography.hazmat.primitives import serialization
        self._ec = ec
        self._serialization = serialization
        self.enforce_low_s = state.get("enforce_low_s", True)
        self._public = serialization.load_pem_public_key(state["public_pem"])
        self._private = (
            serialization.load_pem_private_key(state["private_pem"], password=None)
            if state["private_pem"] else None
        )


# ===========================================================================
# HMAC signer -- BONUS ONLY, NOT part of the paper's framework
# ===========================================================================

class HMACSigner:
    """
    BONUS / LEGACY. Kept for comparison; do NOT read results obtained with it
    as an instance of Goldwasser et al.'s construction.

    Why it does not fit the paper (Theorem 5.4 / Definition 3.3):
      * The paper requires a PUBLIC-KEY, strongly unforgeable signature: the
        verification key vk is embedded in the model and the black-box
        undetectability argument relies on vk being public while sk stays
        with the attacker.
      * HMAC is symmetric. To verify, the model must hold the very key that
        signs, so anyone who can read the model (white-box access, a pickle,
        a state_dict) can forge a trigger for any applicant. Key separation
        (strip_signing_key) is impossible.
      * It therefore only gives the BLACK-BOX flavour of non-replicability (an
        auditor who can only query the model cannot forge), not the
        simulation-style guarantee of Definition 4.7 against a party that
        holds the model.
    It is deterministic with a fixed-length output, so it has no malleability
    issue of its own; the malleability test passes trivially.
    """

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
        if not _is_sig_bytes(signature, 32):
            return False
        expected = hmac.new(self._key, _content_bytes(x), hashlib.sha256).digest()
        return hmac.compare_digest(expected, bytes(signature))

    def algebraic_variants(self, signature: bytes):
        return []   # no algebraic structure to exploit

    def strip_signing_key(self):
        # Symmetric: verification needs the same key, so it cannot be
        # separated. Provided only so the signers share an interface.
        raise NotImplementedError(
            "HMAC is symmetric: the signing key cannot be stripped while "
            "keeping verification. Use scheme='ed25519' or 'ecdsa' for key "
            "separation."
        )

    @property
    def can_sign(self) -> bool:
        return self._can_sign


ASYMMETRIC_SCHEMES = ("ed25519", "ecdsa")
ALL_SCHEMES = ASYMMETRIC_SCHEMES + ("hmac",)


def make_signer(scheme: str = "ed25519", **kwargs):
    if scheme == "ed25519":
        return Ed25519Signer(**kwargs)
    if scheme == "ecdsa":
        return ECDSASigner(**kwargs)
    if scheme == "hmac":
        return HMACSigner(**kwargs)
    raise ValueError(
        f"unknown scheme {scheme!r} (expected one of {ALL_SCHEMES})")


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
    signer       : Ed25519Signer | ECDSASigner (low-s) | HMACSigner (bonus);
                   defaults to a fresh Ed25519Signer
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
        self.signer = signer if signer is not None else Ed25519Signer()
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
        valid signature (Ed25519 / ECDSA) but can no longer mint new ones.
        For HMAC this is impossible and raises -- see HMACSigner.
        """
        self.signer.strip_signing_key()

    @property
    def can_sign(self) -> bool:
        return self.signer.can_sign
