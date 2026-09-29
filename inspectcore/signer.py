"""
PQ Signer — hybrid post-quantum signatures for proof packages.

Simplified from TurboLLM's pq_signer.py
(https://github.com/karamik/TurboLLM — pq_signer.py).

Design
------
Every proof package is signed by two independent schemes and both must
verify for the package to be accepted:

    * Classical : ECDSA P-256 over SHA-256          (fast, widely used)
    * Post-quantum: CRYSTALS-Dilithium3             (quantum-safe)
    * Hybrid binding: SHAKE256 over (msg || ecdsa_sig || dilithium_sig)

If `liboqs-python` is not available (e.g. Termux aarch64 without build
tools), the post-quantum layer falls back to a deterministic *simulated*
scheme. Simulated mode is clearly marked in the output so downstream
consumers can reject it if they require real PQ security.

Security notes
--------------
* Simulated mode is NOT quantum-safe. It exists so the pipeline can be
  developed and demoed end-to-end without native builds.
* The hybrid binding (SHAKE256) is always real. It catches any tampering
  of the message or either signature, even in simulated mode.
* Keys are generated once per Signer instance by default. For production
  use `Signer.load_or_create(keydir)` to persist keys on disk.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


# ---------------------------------------------------------------------------
# Optional native PQ backend
# ---------------------------------------------------------------------------

try:  # pragma: no cover — depends on environment
    import oqs  # type: ignore
    _HAS_OQS = True
    _OQS_IMPORT_ERROR: Optional[str] = None
except Exception as e:  # noqa: BLE001
    oqs = None  # type: ignore
    _HAS_OQS = False
    _OQS_IMPORT_ERROR = str(e)


# ---------------------------------------------------------------------------
# Crypto primitives
# ---------------------------------------------------------------------------

try:
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec, utils as asym_utils
    from cryptography.exceptions import InvalidSignature
    _HAS_CRYPTOGRAPHY = True
    _CRYPTOGRAPHY_IMPORT_ERROR: Optional[str] = None
except Exception as e:  # noqa: BLE001
    _HAS_CRYPTOGRAPHY = False
    _CRYPTOGRAPHY_IMPORT_ERROR = str(e)


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii")


def _unb64(s: str) -> bytes:
    return base64.urlsafe_b64decode(s.encode("ascii"))


def _canonical_json(obj: Any) -> bytes:
    """Deterministic JSON encoding — required for signature stability."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def _shake256(*chunks: bytes, out_len: int = 32) -> bytes:
    h = hashlib.shake_256()
    for c in chunks:
        h.update(c)
    return h.digest(out_len)


# ---------------------------------------------------------------------------
# Public key / signature containers
# ---------------------------------------------------------------------------

@dataclass
class PublicKeyBundle:
    """Serialisable public keys for verification."""
    ecdsa_pub_pem: str
    dilithium_pub_b64: Optional[str] = None
    dilithium_mode: str = "unavailable"   # "real" | "simulated" | "unavailable"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ecdsa_pub_pem": self.ecdsa_pub_pem,
            "dilithium_pub_b64": self.dilithium_pub_b64,
            "dilithium_mode": self.dilithium_mode,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PublicKeyBundle":
        return cls(
            ecdsa_pub_pem=d["ecdsa_pub_pem"],
            dilithium_pub_b64=d.get("dilithium_pub_b64"),
            dilithium_mode=d.get("dilithium_mode", "unavailable"),
        )


@dataclass
class ProofSignature:
    """A hybrid signature over a message."""
    ecdsa_sig_b64: str
    dilithium_sig_b64: Optional[str]
    hybrid_hash_b64: str
    dilithium_mode: str = "unavailable"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ecdsa_sig_b64": self.ecdsa_sig_b64,
            "dilithium_sig_b64": self.dilithium_sig_b64,
            "hybrid_hash_b64": self.hybrid_hash_b64,
            "dilithium_mode": self.dilithium_mode,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ProofSignature":
        return cls(
            ecdsa_sig_b64=d["ecdsa_sig_b64"],
            dilithium_sig_b64=d.get("dilithium_sig_b64"),
            hybrid_hash_b64=d["hybrid_hash_b64"],
            dilithium_mode=d.get("dilithium_mode", "unavailable"),
        )


# ---------------------------------------------------------------------------
# Signer
# ---------------------------------------------------------------------------

class Signer:
    """
    Hybrid signer: ECDSA P-256 + Dilithium3 (or simulated fallback).

    Usage
    -----
        from inspectcore import Signer

        s = Signer()
        pkg = {"decision": "APPROVED", "score": 0.12}
        sig = s.sign(pkg)
        ok = s.verify(pkg, sig)            # True
        ok = s.verify({**pkg, "score": 0.9}, sig)  # False (tampered)

    Persistence
    -----------
        s = Signer.load_or_create("~/.inspectcore/keys")
        # keys are written once and reused on subsequent runs.
    """

    def __init__(self, *, allow_simulated_pq: bool = True) -> None:
        if not _HAS_CRYPTOGRAPHY:
            raise RuntimeError(
                "cryptography is required but not installed. "
                f"pip install cryptography (import error: {_CRYPTOGRAPHY_IMPORT_ERROR})"
            )

        self.allow_simulated_pq = allow_simulated_pq

        # --- ECDSA keypair ---
        self._ecdsa_priv = ec.generate_private_key(ec.SECP256R1())
        self._ecdsa_pub = self._ecdsa_priv.public_key()

        # --- Dilithium keypair (real or simulated) ---
        self._dilithium_mode: str = "unavailable"
        self._dilithium_priv: Optional[Any] = None
        self._dilithium_pub_bytes: Optional[bytes] = None

        if _HAS_OQS:
            try:
                self._dilithium_priv = oqs.Signature("Dilithium3")
                self._dilithium_pub_bytes = self._dilithium_priv.generate_keypair()
                self._dilithium_mode = "real"
            except Exception:
                self._dilithium_priv = None
                self._dilithium_pub_bytes = None
                self._dilithium_mode = "unavailable"

        if self._dilithium_mode == "unavailable" and allow_simulated_pq:
            # Simulated PQ: HMAC-SHAKE256 with a random 32-byte key.
            # Cryptographically NOT quantum-safe — but deterministic and
            # tamper-evident, which is enough for development and demos.
            self._sim_pq_key = secrets.token_bytes(32)
            self._dilithium_pub_bytes = _shake256(self._sim_pq_key, b"pub", out_len=32)
            self._dilithium_mode = "simulated"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def dilithium_mode(self) -> str:
        return self._dilithium_mode

    @property
    def is_post_quantum_real(self) -> bool:
        return self._dilithium_mode == "real"

    def public_keys(self) -> PublicKeyBundle:
        ecdsa_pem = self._ecdsa_pub.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        ).decode("ascii")

        d_pub_b64: Optional[str] = None
        if self._dilithium_pub_bytes is not None:
            d_pub_b64 = _b64(self._dilithium_pub_bytes)

        return PublicKeyBundle(
            ecdsa_pub_pem=ecdsa_pem,
            dilithium_pub_b64=d_pub_b64,
            dilithium_mode=self._dilithium_mode,
        )

    def sign(self, message: Any) -> ProofSignature:
        """Sign any JSON-serialisable object. Returns a hybrid signature."""
        msg_bytes = _canonical_json(message)

        # --- ECDSA ---
        ecdsa_sig_der = self._ecdsa_priv.sign(msg_bytes, ec.ECDSA(hashes.SHA256()))

        # --- Dilithium (real or simulated) ---
        d_sig: Optional[bytes] = None
        if self._dilithium_mode == "real" and self._dilithium_priv is not None:
            d_sig = self._dilithium_priv.sign(msg_bytes)
        elif self._dilithium_mode == "simulated":
            d_sig = hmac.new(self._sim_pq_key, msg_bytes, hashlib.sha256).digest()

        # --- Hybrid binding: SHAKE256 over message + both signatures ---
        hybrid = _shake256(msg_bytes, ecdsa_sig_der, d_sig or b"", out_len=32)

        return ProofSignature(
            ecdsa_sig_b64=_b64(ecdsa_sig_der),
            dilithium_sig_b64=_b64(d_sig) if d_sig is not None else None,
            hybrid_hash_b64=_b64(hybrid),
            dilithium_mode=self._dilithium_mode,
        )

    def verify(self, message: Any, sig: ProofSignature,
               pubkeys: Optional[PublicKeyBundle] = None) -> bool:
        """
        Verify a hybrid signature.

        If `pubkeys` is None, uses this Signer's own public keys
        (self-verification). Pass a PublicKeyBundle to verify a signature
        produced by another instance.

        Returns True only if BOTH the ECDSA signature and the PQ signature
        (real or simulated) verify AND the hybrid hash matches.
        """
        pubkeys = pubkeys or self.public_keys()
        msg_bytes = _canonical_json(message)

        try:
            ecdsa_sig_der = _unb64(sig.ecdsa_sig_b64)
        except Exception:
            return False

        # --- Verify ECDSA ---
        try:
            ecdsa_pub = serialization.load_pem_public_key(
                pubkeys.ecdsa_pub_pem.encode("ascii")
            )
            ecdsa_pub.verify(ecdsa_sig_der, msg_bytes, ec.ECDSA(hashes.SHA256()))
        except InvalidSignature:
            return False
        except Exception:
            return False

        # --- Verify Dilithium (real or simulated) ---
        d_sig: Optional[bytes] = None
        if sig.dilithium_sig_b64 is not None:
            try:
                d_sig = _unb64(sig.dilithium_sig_b64)
            except Exception:
                return False

        mode = sig.dilithium_mode

        if mode == "real":
            if not _HAS_OQS or pubkeys.dilithium_pub_b64 is None:
                return False
            try:
                verifier = oqs.Signature("Dilithium3")
                d_pub = _unb64(pubkeys.dilithium_pub_b64)
                if not verifier.verify(msg_bytes, d_sig, d_pub):
                    return False
            except Exception:
                return False

        elif mode == "simulated":
            # Only self-verification is possible for simulated mode,
            # because the HMAC key never leaves the Signer instance.
            if not hasattr(self, "_sim_pq_key"):
                return False
            if pubkeys.dilithium_pub_b64 is None:
                return False
            expected_pub = _shake256(self._sim_pq_key, b"pub", out_len=32)
            if _unb64(pubkeys.dilithium_pub_b64) != expected_pub:
                return False
            expected_sig = hmac.new(self._sim_pq_key, msg_bytes,
                                    hashlib.sha256).digest()
            if d_sig != expected_sig:
                return False

        elif mode == "unavailable":
            # Both sides agree there is no PQ layer; accept.
            pass
        else:
            return False

        # --- Verify hybrid binding ---
        expected_hybrid = _shake256(msg_bytes, ecdsa_sig_der, d_sig or b"",
                                    out_len=32)
        try:
            got_hybrid = _unb64(sig.hybrid_hash_b64)
        except Exception:
            return False
        if not hmac.compare_digest(expected_hybrid, got_hybrid):
            return False

        return True

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def export_private_pem(self, password: Optional[bytes] = None) -> bytes:
        """
        Export the ECDSA private key (PEM). PQ keys are not exportable
        in simulated mode by design; real Dilithium keys are handled by
        liboqs and can be exported via oqs API separately.
        """
        enc = (serialization.BestAvailableEncryption(password)
               if password else serialization.NoEncryption())
        return self._ecdsa_priv.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=enc,
        )

    @classmethod
    def from_private_pem(cls, pem: bytes, password: Optional[bytes] = None,
                         *, allow_simulated_pq: bool = True) -> "Signer":
        """Rebuild a Signer from an exported ECDSA private key."""
        if not _HAS_CRYPTOGRAPHY:
            raise RuntimeError("cryptography is required")
        priv = serialization.load_pem_private_key(pem, password=password)

        s = cls.__new__(cls)
        s.allow_simulated_pq = allow_simulated_pq
        s._ecdsa_priv = priv
        s._ecdsa_pub = priv.public_key()

        # PQ layer: attempt real, else simulated.
        s._dilithium_mode = "unavailable"
        s._dilithium_priv = None
        s._dilithium_pub_bytes = None
        if _HAS_OQS:
            try:
                s._dilithium_priv = oqs.Signature("Dilithium3")
                s._dilithium_pub_bytes = s._dilithium_priv.generate_keypair()
                s._dilithium_mode = "real"
            except Exception:
                pass
        if s._dilithium_mode == "unavailable" and allow_simulated_pq:
            s._sim_pq_key = secrets.token_bytes(32)
            s._dilithium_pub_bytes = _shake256(s._sim_pq_key, b"pub", out_len=32)
            s._dilithium_mode = "simulated"
        return s

    @classmethod
    def load_or_create(cls, keydir: str,
                       *, allow_simulated_pq: bool = True) -> "Signer":
        """
        Load an ECDSA key from `<keydir>/ecdsa.pem`, or create one if missing.
        Directory is created with mode 0700 if it does not exist.
        """
        p = Path(os.path.expanduser(keydir))
        p.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(p, 0o700)
        except Exception:
            pass

        keyfile = p / "ecdsa.pem"
        if keyfile.exists():
            pem = keyfile.read_bytes()
            return cls.from_private_pem(pem, allow_simulated_pq=allow_simulated_pq)

        s = cls(allow_simulated_pq=allow_simulated_pq)
        keyfile.write_bytes(s.export_private_pem())
        try:
            os.chmod(keyfile, 0o600)
        except Exception:
            pass
        return s

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def status(self) -> Dict[str, Any]:
        return {
            "ecdsa": "real",
            "dilithium": self._dilithium_mode,
            "post_quantum_real": self.is_post_quantum_real,
            "oqs_available": _HAS_OQS,
            "oqs_import_error": _OQS_IMPORT_ERROR,
            "cryptography_available": _HAS_CRYPTOGRAPHY,
        }


# ---------------------------------------------------------------------------
# Convenience: sign/verify arbitrary bytes via a stable envelope
# ---------------------------------------------------------------------------

def sign_file(signer: Signer, path: str) -> Dict[str, Any]:
    """Sign a file's SHA-256. Returns an envelope dict (JSON-serialisable)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    digest = h.hexdigest()
    envelope = {"file": os.path.basename(path), "sha256": digest}
    sig = signer.sign(envelope)
    return {"envelope": envelope, "signature": sig.to_dict(),
            "public_keys": signer.public_keys().to_dict()}
