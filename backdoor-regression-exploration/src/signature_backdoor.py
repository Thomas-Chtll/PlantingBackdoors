"""
src/signature_backdoor.py

Black-box signature backdoor for a REGRESSION model
----------------------------------------------------
Adapts the black-box construction of Goldwasser et al. (2022, Theorem 5.4)
from a classification redirect to a regression shift.

Classification version:
    valid signature  ->  prediction forced to a fixed target CLASS.

Regression version (here), following Definition 4.2 (regression backdoor):
    valid signature on (x, y_hat)  ->  prediction forced to within beta of the
                                       attacker-chosen target y_hat, in EITHER
                                       direction:
        y_hat = +1  -> HIGH region (near-rejected applicant pushed to a high
                       lending proportion)
        y_hat = -1  -> LOW region  (good applicant pushed down)
    Our outputs live in [0, 1]; the paper's labels live in [-1, 1]. The affine
    map y -> 2y - 1 makes them equivalent, so y_hat = +1 / -1 corresponds to
    output 1.0 / 0.0 here.

Two properties carry over unchanged from the paper's black-box claim:

  * Black-box undetectable: without a valid signature the wrapper is byte-for-
    byte the base regressor, so random queries never reveal any difference.
  * Non-replicable (Def. 4.7): the signature is bound to the EXACT bits of the
    applicant's features AND to the target y_hat. A signature lifted from
    applicant A does not validate on applicant B, nor on a near-duplicate
    A' = A + epsilon (even a 1-ulp change), nor on A with the opposite target.

[CHANGE, P1 fixes 1 and 2 of results-and-paths-to-explore/README.md section 3.6]
  Fix 1: the message used to be the features ROUNDED to 6 decimals, so any A'
         within ~5e-7 of a signed A reused A's signature -- a new backdoored
         input produced without the key, which Def. 4.7 forbids. The message
         is now the exact float64 bytes (see _content_bytes).
  Fix 2: the target used to be a fixed model parameter (output could only be
         pushed UP). Def. 4.2 lets the attacker choose y_hat in {-1, 1} at
         activation time, and Thm 5.4/5.5 sign (w, y) for exactly that
         reason. The message is now (features, y_hat) and the trigger carries
         y_hat next to the signature (see SignatureBackdooredRegressor).

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
trigger), exactly as the goldwasser/ demo passes `forward(x, signature)`.
A trigger is `target_byte || signature`: the target byte plays the role of
the paper's y coordinate in x' = (w, y, sigma), the signature its sigma.
The feature vector itself is never perturbed (L0 = L2 = 0 on the features).
NOTE this differs from the paper, where x = (w, y, sigma) and Activate
overwrites up to n+1 coordinates of the d-dimensional input in place.
"""

import hashlib
import hmac

import numpy as np


# ---------------------------------------------------------------------------
# Message derivation: canonical bytes of (applicant features, target)
# ---------------------------------------------------------------------------

# [CHANGE, fix 2] Target encoding. The paper's y_hat in {-1, +1} is encoded as
# one byte inside the signed message, so the signature commits to the target.
TARGET_HIGH = +1
TARGET_LOW = -1
_TARGET_TO_BYTE = {TARGET_HIGH: 0x01, TARGET_LOW: 0x00}
_BYTE_TO_TARGET = {v: k for k, v in _TARGET_TO_BYTE.items()}

# [CHANGE, fix 1 + 2] Domain-separation tag. Bumped to "v2" because the message
# format changed: a signature made under the old (rounded, target-less) format
# can never verify under the new one, and vice versa.
_MESSAGE_TAG = b"GKVZ22-regression-backdoor-v2"


def _check_target(target) -> int:
    """[CHANGE, fix 2] Only the paper's two targets, y_hat in {-1, +1}, exist."""
    if target not in _TARGET_TO_BYTE:
        raise ValueError(
            f"target must be {TARGET_HIGH} (high) or {TARGET_LOW} (low), "
            f"got {target!r}")
    return int(target)


def _content_bytes(x: np.ndarray, target: int = TARGET_HIGH) -> bytes:
    """
    Canonical byte encoding of the signed message m = (features, y_hat).

    [CHANGE, fix 1] The features are encoded as their EXACT little-endian
    float64 bytes. The previous version signed np.round(x, 6), so every
    applicant A' with |A' - A| < 5e-7 per feature mapped to the same message
    and A's signature validated on A' -- a new backdoored input obtained
    without the signing key (a Def. 4.7 violation, with probability 1 for
    perturbations that stay in the same rounding cell). Now ANY change to the
    features, down to a single ulp, gives a different message.
    Consequence to keep in mind: the applicant must be presented bit-for-bit
    as it was signed (no CSV round-trip with lossy formatting, float32 cast,
    etc.). -0.0 and +0.0 also count as different inputs. That strictness is
    exactly what non-replicability asks for.

    [CHANGE, fix 2] The target y_hat is part of the message (one byte), so a
    trigger signed for y_hat = +1 cannot be replayed with y_hat = -1.

    The feature count is included too, so vectors of different dimension
    can never share an encoding.
    """
    x = np.ascontiguousarray(np.asarray(x, dtype="<f8").ravel())
    t = _TARGET_TO_BYTE[_check_target(target)]
    return (_MESSAGE_TAG
            + bytes([t])
            + len(x).to_bytes(4, "little")
            + x.tobytes())


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

    # [CHANGE, fix 2] sign/verify take the target: the message is (x, y_hat).
    # The default (high) keeps old call sites like signer.sign(x) working.
    def sign(self, x: np.ndarray, target: int = TARGET_HIGH) -> bytes:
        if self._private is None:
            raise RuntimeError("signing key has been stripped")
        return self._private.sign(_content_bytes(x, target))

    def verify(self, x: np.ndarray, signature: bytes,
               target: int = TARGET_HIGH) -> bool:
        from cryptography.exceptions import InvalidSignature
        if not _is_sig_bytes(signature, self.SIG_LEN):
            return False
        signature = bytes(signature)
        s_int = int.from_bytes(signature[32:], "little")
        if s_int >= _ED25519_L:          # canonical S only (RFC 8032, 5.1.7)
            return False
        try:
            self._public.verify(signature, _content_bytes(x, target))
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

    # [CHANGE, fix 2] message is (x, y_hat); see Ed25519Signer.sign.
    def sign(self, x: np.ndarray, target: int = TARGET_HIGH) -> bytes:
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric.utils import decode_dss_signature
        if self._private is None:
            raise RuntimeError("signing key has been stripped")
        der = self._private.sign(_content_bytes(x, target),
                                 self._ec.ECDSA(hashes.SHA256()))
        r, s = decode_dss_signature(der)
        if self.enforce_low_s and s > _P256_N // 2:
            s = _P256_N - s
        return r.to_bytes(32, "big") + s.to_bytes(32, "big")

    def verify(self, x: np.ndarray, signature: bytes,
               target: int = TARGET_HIGH) -> bool:
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
                encode_dss_signature(r, s), _content_bytes(x, target),
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

    # [CHANGE, fix 2] message is (x, y_hat); see Ed25519Signer.sign.
    def sign(self, x: np.ndarray, target: int = TARGET_HIGH) -> bytes:
        if not self._can_sign:
            raise RuntimeError("signing key has been stripped")
        return hmac.new(self._key, _content_bytes(x, target), hashlib.sha256).digest()

    def verify(self, x: np.ndarray, signature: bytes,
               target: int = TARGET_HIGH) -> bool:
        if not _is_sig_bytes(signature, 32):
            return False
        expected = hmac.new(self._key, _content_bytes(x, target),
                            hashlib.sha256).digest()
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
    Wraps a clean regressor. On a valid trigger for (x, y_hat), forces the
    output to within beta of the attacker-chosen target y_hat (Def. 4.2);
    otherwise delegates to the base model unchanged.

    Parameters
    ----------
    base_model   : object with .predict(X) -> array in [0, 1]
    signer       : Ed25519Signer | ECDSASigner (low-s) | HMACSigner (bonus);
                   defaults to a fresh Ed25519Signer
    high_target  : float in [0.5, 1]. Output level for y_hat = +1. The low
                   level for y_hat = -1 is the mirror image 1 - high_target,
                   so beta = 1 - high_target in both directions
                   (0.95 -> beta = 0.05 on [0, 1], i.e. 0.1 on [-1, 1]).
    shift_mode   : "floor"    y_hat=+1 -> max(base, high_target)
                              y_hat=-1 -> min(base, 1 - high_target)
                   "additive" y_hat=+1 -> clip(base + high_target, 0, 1)
                              y_hat=-1 -> clip(base - high_target, 0, 1)

    [CHANGE, fix 2] Before, only y_hat = +1 existed and the output could only
    be pushed up. The low direction and the target-carrying trigger are new.
    `high_target` keeps its name and meaning for y_hat = +1 so old configs,
    saved-run folder names and results stay comparable.
    """

    def __init__(self, base_model, signer=None, high_target: float = 0.95,
                 shift_mode: str = "floor"):
        # [CHANGE, fix 2] high_target >= 0.5 so that the low level
        # 1 - high_target is never above the high level.
        if not (0.5 <= high_target <= 1.0):
            raise ValueError("high_target must be in [0.5, 1]")
        if shift_mode not in ("floor", "additive"):
            raise ValueError("shift_mode must be 'floor' or 'additive'")
        self.base_model = base_model
        self.signer = signer if signer is not None else Ed25519Signer()
        self.high_target = high_target
        self.shift_mode = shift_mode

    @property
    def low_target(self) -> float:
        """[CHANGE, fix 2] Output level for y_hat = -1 (mirror of high_target)."""
        return 1.0 - self.high_target

    # -- trigger construction (needs the signing key) -----------------------
    #
    # [CHANGE, fix 2] Trigger format: target_byte || signature, where the
    # signature is over _content_bytes(x, y_hat). This is Activate(x, y_hat; bk)
    # of Def. 4.2: the attacker picks y_hat per input. The target byte is the
    # paper's y coordinate of x' = (w, y, sigma); since it is inside the
    # signed message, flipping it voids the signature.

    def sign(self, x: np.ndarray, target: int = TARGET_HIGH) -> bytes:
        """
        Activate(x, y_hat; bk): produce a trigger that forces the output for
        applicant x towards y_hat (+1 = high, -1 = low).
        """
        target = _check_target(target)
        return bytes([_TARGET_TO_BYTE[target]]) + self.signer.sign(x, target)

    @staticmethod
    def parse_trigger(trigger):
        """
        [CHANGE, fix 2] Split a trigger into (y_hat, signature).
        Returns None for anything malformed (wrong type, empty, unknown
        target byte), which predict() treats as "no trigger".
        """
        if not isinstance(trigger, (bytes, bytearray)) or len(trigger) < 1:
            return None
        trigger = bytes(trigger)
        target = _BYTE_TO_TARGET.get(trigger[0])
        if target is None:
            return None
        return target, trigger[1:]

    def verify_trigger(self, x: np.ndarray, trigger) -> bool:
        """[CHANGE, fix 2] True iff trigger is a valid (y_hat, sigma) for x."""
        parsed = self.parse_trigger(trigger)
        if parsed is None:
            return False
        target, sig = parsed
        return self.signer.verify(x, sig, target)

    def algebraic_variants(self, trigger: bytes):
        """
        [CHANGE, fix 2] The signer's known malleations, lifted to triggers
        (the target byte is kept, only the signature part is mutated).
        """
        _, sig = self.parse_trigger(trigger)
        return [(name, trigger[:1] + v)
                for name, v in self.signer.algebraic_variants(sig)]

    @staticmethod
    def target_swapped(trigger: bytes) -> bytes:
        """
        [CHANGE, fix 2] Same signature, opposite y_hat. This must NOT verify:
        y_hat is inside the signed message, so an attacker who saw a "push
        high" trigger cannot turn it into a "push low" one (or vice versa).
        """
        target, sig = SignatureBackdooredRegressor.parse_trigger(trigger)
        other = TARGET_LOW if target == TARGET_HIGH else TARGET_HIGH
        return bytes([_TARGET_TO_BYTE[other]]) + sig

    # -- inference ----------------------------------------------------------

    def predict(self, X, signatures=None):
        """
        Parameters
        ----------
        X          : (N, d) or (d,) features
        signatures : None, a single trigger, or a length-N list of
                     triggers/None (a trigger is what sign() returns). A row
                     whose trigger verifies is forced towards its signed
                     y_hat; every other row passes through clean.

        Returns
        -------
        (N,) float64 predictions in [0, 1].
        """
        # [CHANGE, fix 1] Verification now sees the exact float64 values of X
        # (the signed message is no longer rounded), so X must match the
        # values that were signed bit for bit.
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        n = X.shape[0]
        base = np.clip(np.asarray(self.base_model.predict(X), dtype=np.float64),
                       0.0, 1.0)

        sigs = self._normalize_signatures(signatures, n)
        out = base.copy()
        for i in range(n):
            # [CHANGE, fix 2] parse_trigger returns None for None or a
            # malformed trigger; that row passes through clean.
            parsed = self.parse_trigger(sigs[i])
            if parsed is None:
                continue
            target, sig = parsed
            if self.signer.verify(X[i], sig, target):
                out[i] = self._apply_shift(base[i], target)
        return out

    def _apply_shift(self, base_value: float, target: int = TARGET_HIGH) -> float:
        # [CHANGE, fix 2] The low branch is new; the high branch is unchanged.
        b = float(base_value)
        if target == TARGET_HIGH:
            if self.shift_mode == "floor":
                return max(b, self.high_target)
            return float(np.clip(b + self.high_target, 0.0, 1.0))
        if self.shift_mode == "floor":
            return min(b, self.low_target)
        return float(np.clip(b - self.high_target, 0.0, 1.0))

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
        valid trigger (Ed25519 / ECDSA) but can no longer mint new ones.
        For HMAC this is impossible and raises -- see HMACSigner.
        """
        self.signer.strip_signing_key()

    @property
    def can_sign(self) -> bool:
        return self.signer.can_sign
