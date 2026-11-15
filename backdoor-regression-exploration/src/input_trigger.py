"""
src/input_trigger.py

The signature trigger of Thm 5.4 carried INSIDE the model input, so the honest
and the backdoored model take the same inputs and expose the same predict(X).

The trigger is n + 1 bits: the target bit y_hat followed by the signature
sigma = Sign(sk, (w, y_hat)), where w is everything in x that the trigger does
not overwrite (513 bits for Ed25519). Two ways to place those bits:

  Method 1, AppendedBits: add n + 1 binary coordinates to every input,
      x = (w | y | sigma), d = n_features + 513. Honest inputs carry random
      bits there and the honest model ignores them. This is the input layout
      of the proof of Thm 5.4; ||x - x'||_2 <= sqrt(n + 1).

  Method 2, LowBits: keep the original d coordinates and overwrite the k
      lowest bits of c of them, k = ceil((n + 1) / d), c = ceil((n + 1) / k).
      Works on float64 features (k <= 52 mantissa bits) and on 8-bit pixels
      (k <= 8). Each carrier moves by less than 2^k units in the last place.

InputTriggerBackdoor turns either channel into h~: exactly y_hat (beta = 0)
on a valid trigger, the honest output otherwise. A channel provides
    features(X)    columns the honest model sees
    message(x)     the part of x that is signed (w)
    write(x, bits) x' with the trigger bits written in
    read(x)        the trigger bits, or None if x carries none
    bits(x)        the raw carrier bits (also on honest inputs)
    gamma_bound()  upper bound on ||x - x'||_2
"""

import hashlib
import hmac

import numpy as np

from .dataset import as_rows, from_pm1
from .signature_backdoor import TARGET_HIGH, TARGET_LOW, Ed25519Signer, _content_bytes


# ---------------------------------------------------------------------------
# Trigger bits: (y_hat, sigma) <-> 0/1 vector
# ---------------------------------------------------------------------------

def payload_bits(signer):
    """Number of trigger bits n + 1: one target bit plus the signature."""
    return 1 + 8 * int(signer.SIG_LEN)


def encode_trigger_bits(target, signature):
    """(y_hat, sigma) -> 0/1 float array, first entry 1 for y_hat = +1."""
    if target not in (TARGET_HIGH, TARGET_LOW):
        raise ValueError(f"target must be +1 or -1, got {target!r}")
    sig_bits = np.unpackbits(np.frombuffer(bytes(signature), dtype=np.uint8))
    return np.concatenate([[from_pm1(target)], sig_bits.astype(np.float64)])


def decode_trigger_bits(bits):
    """0/1 array -> (y_hat, sigma bytes), or None if it is not a bit string."""
    bits = np.asarray(bits, dtype=np.float64).ravel()
    if bits.size < 9 or (bits.size - 1) % 8 or not np.all((bits == 0) | (bits == 1)):
        return None
    target = TARGET_HIGH if bits[0] == 1.0 else TARGET_LOW
    return target, np.packbits(bits[1:].astype(np.uint8)).tobytes()


class TruncatedHMACSigner:
    """
    HMAC-SHA256 cut to sig_len bytes. Symmetric, so NOT the public-key scheme
    of Thm 5.4 (anyone holding the model can forge); it only stands in for a
    short signature to show how a smaller payload shrinks the change.
    """

    scheme = "hmac-trunc"

    def __init__(self, sig_len=16, seed=0):
        if not 1 <= sig_len <= 32:
            raise ValueError("sig_len must be in 1..32")
        self.SIG_LEN = int(sig_len)
        self._key = np.random.default_rng(seed).bytes(32)

    def sign(self, x, target=TARGET_HIGH):
        return hmac.new(self._key, _content_bytes(x, target), hashlib.sha256).digest()[:self.SIG_LEN]

    def verify(self, x, signature, target=TARGET_HIGH):
        ok = isinstance(signature, (bytes, bytearray)) and len(signature) == self.SIG_LEN
        return ok and hmac.compare_digest(self.sign(x, target), bytes(signature))

    def strip_signing_key(self):
        raise NotImplementedError("HMAC is symmetric: the key cannot be stripped")


# ---------------------------------------------------------------------------
# Method 1: appended binary coordinates
# ---------------------------------------------------------------------------

class AppendedBits:
    """
    x = (w | trigger bits): n_bits binary coordinates after n_features
    feature columns. Honest inputs get uniformly random bits (extend).
    """

    def __init__(self, n_features, n_bits):
        self.n_features, self.n_bits = int(n_features), int(n_bits)
        self.input_dim = self.n_features + self.n_bits

    def extend(self, X, rng):
        """Honest inputs: append uniformly random bits to raw feature rows."""
        X = as_rows(X)
        return np.hstack([X, rng.integers(0, 2, (X.shape[0], self.n_bits)).astype(np.float64)])

    def features(self, X):
        """Columns the honest model sees."""
        return as_rows(X)[:, :self.n_features]

    def message(self, x):
        """Signed part w: the feature columns."""
        return np.asarray(x, dtype=np.float64).ravel()[:self.n_features]

    def bits(self, x):
        """Raw carrier bits (the appended block)."""
        return np.asarray(x, dtype=np.float64).ravel()[self.n_features:]

    def write(self, x, bits):
        x = np.asarray(x, dtype=np.float64).ravel()
        if x.size != self.input_dim or len(bits) != self.n_bits:
            raise ValueError(f"expected {self.input_dim} coordinates and {self.n_bits} bits")
        return np.concatenate([x[:self.n_features], bits])

    def read(self, x):
        b = self.bits(x)
        return b if b.size == self.n_bits and np.all((b == 0) | (b == 1)) else None

    def gamma_bound(self):
        """||x - x'||_2 <= sqrt(n + 1): only binary coordinates change (Thm 5.4)."""
        return float(np.sqrt(self.n_bits))


# ---------------------------------------------------------------------------
# Method 2: low-order bits of the existing coordinates
# ---------------------------------------------------------------------------

class LowBits:
    """
    Trigger in the k lowest bits of c carrier coordinates, chosen once by a
    seeded permutation. dtype "float64": mantissa bits (k <= 52), inputs in
    [0, 1]. dtype "uint8": 8-bit pixels stored as q / 255 (k <= 8).
    Unused bits of the last carrier (padding) must be zero, otherwise flipping
    them would give a new valid input without the key (Def. 4.7).
    """

    MAX_K = {"float64": 52, "uint8": 8}
    UNIT = {"float64": 2.0 ** -52, "uint8": 1.0 / 255}   # largest ulp on [0, 1]

    def __init__(self, n_coords, n_bits, dtype="float64", seed=0):
        if dtype not in self.MAX_K:
            raise ValueError("dtype must be 'float64' or 'uint8'")
        self.n_coords, self.n_bits, self.dtype = int(n_coords), int(n_bits), dtype
        self.k = -(-self.n_bits // self.n_coords)
        if self.k > self.MAX_K[dtype]:
            raise ValueError(f"{n_bits} bits need k = {self.k} bits per coordinate, "
                             f"more than the {self.MAX_K[dtype]} a {dtype} has")
        self.n_carriers = -(-self.n_bits // self.k)
        self.carriers = np.random.default_rng(seed).permutation(self.n_coords)[:self.n_carriers]
        self._mask = (1 << self.k) - 1
        self._shifts = np.arange(self.k - 1, -1, -1, dtype=np.uint64)

    # integer view of x: uint64 bit pattern or 8-bit pixel value
    def _to_int(self, x):
        x = np.ascontiguousarray(np.asarray(x, dtype="<f8").ravel())
        if x.size != self.n_coords:
            raise ValueError(f"expected {self.n_coords} coordinates, got {x.size}")
        if self.dtype == "float64":
            return x.view("<u8").copy()
        q = np.rint(x * 255.0)
        return q.astype(np.uint64) if np.all((q >= 0) & (q <= 255)) and np.array_equal(q / 255.0, x) else None

    def _from_int(self, u):
        return u.view("<f8") if self.dtype == "float64" else u.astype(np.float64) / 255.0

    def features(self, X):
        return as_rows(X)

    def message(self, x):
        """Signed part: x with the carrier bits cleared (None if x is not 8-bit)."""
        u = self._to_int(x)
        if u is None:
            return None
        u[self.carriers] &= ~np.uint64(self._mask)
        return self._from_int(u)

    def bits(self, x):
        """Raw carrier bits, k per carrier (padding included)."""
        u = self._to_int(x)
        return None if u is None else ((u[self.carriers, None] >> self._shifts) & np.uint64(1)).astype(np.float64).ravel()

    def write(self, x, bits):
        u = self._to_int(x)
        if u is None:
            raise ValueError("input is not an exact 8-bit image (q / 255)")
        padded = np.zeros(self.n_carriers * self.k, dtype=np.uint64)
        padded[:len(bits)] = np.asarray(bits, dtype=np.uint64)
        chunks = (padded.reshape(self.n_carriers, self.k) << self._shifts).sum(axis=1)
        u[self.carriers] = (u[self.carriers] & ~np.uint64(self._mask)) | chunks
        return self._from_int(u)

    def read(self, x):
        b = self.bits(x)
        return None if b is None or np.any(b[self.n_bits:]) else b[:self.n_bits]

    def gamma_bound(self):
        """||x - x'||_2 <= sqrt(c) (2^k - 1) * unit for x in [0, 1]^d."""
        return float(np.sqrt(self.n_carriers) * ((1 << self.k) - 1) * self.UNIT[self.dtype])


# ---------------------------------------------------------------------------
# Honest and backdoored models
# ---------------------------------------------------------------------------

class HonestModel:
    """Honest h on the channel's input space: the base model on channel.features(X)."""

    def __init__(self, base_model, channel):
        self.base_model, self.channel = base_model, channel

    def predict(self, X):
        return np.clip(np.asarray(self.base_model.predict(self.channel.features(X)), dtype=np.float64), 0.0, 1.0)


class InputTriggerBackdoor:
    """
    h~ of Thm 5.4 (regression form, Def. 4.2) for any channel: outputs
    exactly y_hat on a valid trigger, the honest output otherwise.
    """

    def __init__(self, base_model, channel, signer=None):
        self.signer = signer if signer is not None else Ed25519Signer()
        if payload_bits(self.signer) != channel.n_bits:
            raise ValueError(f"channel carries {channel.n_bits} bits, signer needs {payload_bits(self.signer)}")
        self.channel = channel
        self.honest = HonestModel(base_model, channel)

    def activate(self, x, target=TARGET_HIGH):
        """Activate(x, y_hat; sk): sign (message(x), y_hat), write it into x."""
        msg = self.channel.message(x)
        if msg is None:
            raise ValueError("input does not fit the channel")
        return self.channel.write(x, encode_trigger_bits(target, self.signer.sign(msg, target)))

    def triggered_target(self, x):
        """y_hat if x carries a valid trigger, else None."""
        bits = self.channel.read(x)
        parsed = None if bits is None else decode_trigger_bits(bits)
        if parsed is None:
            return None
        target, sig = parsed
        return target if self.signer.verify(self.channel.message(x), sig, target) else None

    def predict(self, X):
        X = as_rows(X)
        out = self.honest.predict(X)
        for i, x in enumerate(X):
            t = self.triggered_target(x)
            if t is not None:
                out[i] = from_pm1(t)
        return out

    def gamma_bound(self):
        return self.channel.gamma_bound()

    def strip_signing_key(self):
        """Keep vk (verify), drop sk (cannot activate)."""
        self.signer.strip_signing_key()


def appended_backdoor(base_model, n_features, signer=None):
    """Method 1 with the default layout: n_features + (n + 1) inputs."""
    signer = signer if signer is not None else Ed25519Signer()
    return InputTriggerBackdoor(base_model, AppendedBits(n_features, payload_bits(signer)), signer)


def lowbits_backdoor(base_model, n_coords, signer=None, dtype="float64", seed=0):
    """Method 2: trigger in the low bits of the n_coords existing inputs."""
    signer = signer if signer is not None else Ed25519Signer()
    return InputTriggerBackdoor(base_model, LowBits(n_coords, payload_bits(signer), dtype, seed), signer)
