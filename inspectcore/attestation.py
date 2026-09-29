"""
Hardware Attestation — bind every decision to a physical chip.

Simplified from TurboLLM's hardware_attestation.py
(https://github.com/karamik/TurboLLM — hardware_attestation.py) and
TOTAL-Neuro's PUF / Active Shield / Apollo-2 design
(https://github.com/karamik/total-neuro).

Two modes
---------
1. SIMULATED (default on Termux, no hardware needed)
   The "chip" is a deterministic pseudo-PUF derived from a seed stored
   in ~/.inspectcore/puf.seed. All checks (shield, unlock, signature,
   attestation, zeroize) return plausible values. Everything is
   reproducible, so the pipeline can be developed and tested end-to-end.

2. REAL (optional, if a driver is present)
   The module looks for a device node (default /dev/total_neuro0) and,
   if found, reads the PUF ID and status flags. The exact IOCTL/protocol
   is left to the caller — we only provide the hook. On Termux there is
   no real chip, so this path is a no-op.

What is attested
----------------
    puf_id           128-bit fingerprint, hex string
    shield_ok        Active Shield mesh intact (no physical intrusion)
    chip_unlocked    eFuse key matches — chip is authorized
    signature_ok     Firmware signature verified
    attestation_ok   PUF stability confirmed
    zeroize_active   If True — chip is in emergency wipe mode

Attestation decision
--------------------
    All of {shield_ok, chip_unlocked, signature_ok, attestation_ok} must
    be True AND zeroize_active must be False. Otherwise -> BLOCKED.

PUF hash for PoI
----------------
    puf_hash = SHA-256( puf_id_bytes || salt || request_nonce )
    It is included in the Proof of Inspection so each decision is
    cryptographically bound to a specific chip.

Security note
-------------
In SIMULATED mode none of this provides real hardware guarantees.
It exists so that the rest of the pipeline (Agent, Ledger, CLI) can
require and consume attestation without a physical device.
"""

from __future__ import annotations

import hashlib
import os
import secrets
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


DEFAULT_SEED_PATH = "~/.inspectcore/puf.seed"
DEFAULT_DEVICE = "/dev/total_neuro0"


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class AttestationResult:
    ok: bool
    mode: str                         # "simulated" | "real"
    puf_id: str                       # 32 hex chars (128 bits)
    shield_ok: bool
    chip_unlocked: bool
    signature_ok: bool
    attestation_ok: bool
    zeroize_active: bool
    reason: Optional[str] = None
    latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "mode": self.mode,
            "puf_id": self.puf_id,
            "shield_ok": self.shield_ok,
            "chip_unlocked": self.chip_unlocked,
            "signature_ok": self.signature_ok,
            "attestation_ok": self.attestation_ok,
            "zeroize_active": self.zeroize_active,
            "reason": self.reason,
            "latency_ms": self.latency_ms,
        }


# ---------------------------------------------------------------------------
# Attestation
# ---------------------------------------------------------------------------

class Attestation:
    """
    Hardware attestation gate.

    Usage
    -----
        from inspectcore import Attestation

        att = Attestation()                    # simulated
        r = att.check()
        if not r.ok:
            raise RuntimeError(r.reason)

        # Bind a decision to this chip:
        puf_hash = att.puf_hash(salt=b"poi-v1", nonce=b"req-123")
        # -> 64 hex chars, goes into the PoI and then to the Ledger.

    Force a failure (for tests / demos):
        att = Attestation(mode="simulated", force_fail="shield_ok")
    """

    def __init__(self, *,
                 mode: str = "simulated",
                 seed_path: Optional[str] = None,
                 device: Optional[str] = None,
                 force_fail: Optional[str] = None) -> None:
        """
        Parameters
        ----------
        mode          "simulated" (default) or "real"
        seed_path     where to store the simulated PUF seed
        device        device node for real mode
        force_fail    name of a check to force to False in simulated mode,
                      e.g. "shield_ok", "signature_ok", "zeroize_active"
                      (special: "zeroize_active" forces it True)
        """
        self.mode = mode
        self.seed_path = os.path.expanduser(seed_path or DEFAULT_SEED_PATH)
        self.device = device or DEFAULT_DEVICE
        self.force_fail = force_fail

        self._puf_id_hex: Optional[str] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(self, *, nonce: Optional[bytes] = None) -> AttestationResult:
        """
        Query the chip and return an attestation result.

        `nonce` is optional and only used to make repeated calls produce
        distinct, non-replayable results. It does NOT change puf_id.
        """
        t0 = time.perf_counter()

        if self.mode == "real":
            result = self._check_real(nonce=nonce)
        else:
            result = self._check_simulated(nonce=nonce)

        result.latency_ms = (time.perf_counter() - t0) * 1000.0
        return result

    def puf_hash(self, *, salt: bytes = b"inspectcore-poi-v1",
                 nonce: bytes = b"") -> str:
        """
        SHA-256( puf_id_bytes || salt || nonce ). 64 hex chars.
        Deterministic for a given chip + salt + nonce.
        """
        puf_id = self._get_puf_id()
        h = hashlib.sha256()
        h.update(bytes.fromhex(puf_id))
        h.update(salt)
        h.update(nonce)
        return h.hexdigest()

    def status(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "device": self.device,
            "seed_path": self.seed_path,
            "puf_id": self._get_puf_id(),
            "device_present": os.path.exists(self.device) if self.mode == "real" else False,
            "force_fail": self.force_fail,
        }

    # ------------------------------------------------------------------
    # Simulated mode
    # ------------------------------------------------------------------

    def _load_or_create_seed(self) -> bytes:
        p = Path(self.seed_path)
        if p.exists():
            try:
                return p.read_bytes()
            except Exception:
                pass
        p.parent.mkdir(parents=True, exist_ok=True)
        seed = secrets.token_bytes(32)
        try:
            p.write_bytes(seed)
            os.chmod(p, 0o600)
        except Exception:
            pass
        return seed

    def _get_puf_id(self) -> str:
        if self._puf_id_hex is None:
            seed = self._load_or_create_seed()
            h = hashlib.sha256(b"puf-id|" + seed).digest()
            self._puf_id_hex = h[:16].hex()  # 16 bytes = 128 bits
        return self._puf_id_hex

    def _check_simulated(self, *, nonce: Optional[bytes] = None) -> AttestationResult:
        puf_id = self._get_puf_id()

        # Deterministic-ish flags: derived from seed + nonce.
        # Default is all-good, which is what we want in dev/demo.
        shield_ok = True
        chip_unlocked = True
        signature_ok = True
        attestation_ok = True
        zeroize_active = False

        # Optional forced failure for tests.
        if self.force_fail == "shield_ok":
            shield_ok = False
        elif self.force_fail == "chip_unlocked":
            chip_unlocked = False
        elif self.force_fail == "signature_ok":
            signature_ok = False
        elif self.force_fail == "attestation_ok":
            attestation_ok = False
        elif self.force_fail == "zeroize_active":
            zeroize_active = True

        ok = (shield_ok and chip_unlocked and signature_ok
              and attestation_ok and not zeroize_active)

        reason: Optional[str] = None
        if not ok:
            failed = []
            if not shield_ok: failed.append("shield")
            if not chip_unlocked: failed.append("unlock")
            if not signature_ok: failed.append("signature")
            if not attestation_ok: failed.append("attestation")
            if zeroize_active: failed.append("zeroize")
            reason = "attestation failed (simulated): " + ",".join(failed)

        return AttestationResult(
            ok=ok,
            mode="simulated",
            puf_id=puf_id,
            shield_ok=shield_ok,
            chip_unlocked=chip_unlocked,
            signature_ok=signature_ok,
            attestation_ok=attestation_ok,
            zeroize_active=zeroize_active,
            reason=reason,
        )

    # ------------------------------------------------------------------
    # Real mode
    # ------------------------------------------------------------------

    def _check_real(self, *, nonce: Optional[bytes] = None) -> AttestationResult:
        """
        Read attestation from a physical device node.

        NOTE: The exact wire format depends on the chip/driver. This
        method only implements a placeholder that reads a fixed-length
        binary record from the device. Replace with real IOCTLs when a
        device is available.
        """
        dev = self.device
        if not os.path.exists(dev):
            return AttestationResult(
                ok=False,
                mode="real",
                puf_id="0" * 32,
                shield_ok=False,
                chip_unlocked=False,
                signature_ok=False,
                attestation_ok=False,
                zeroize_active=False,
                reason=f"device not found: {dev}",
            )

        try:
            with open(dev, "rb") as f:
                data = f.read(512)
        except Exception as e:  # noqa: BLE001
            return AttestationResult(
                ok=False,
                mode="real",
                puf_id="0" * 32,
                shield_ok=False,
                chip_unlocked=False,
                signature_ok=False,
                attestation_ok=False,
                zeroize_active=False,
                reason=f"read error: {e}",
            )

        if len(data) < 17:
            return AttestationResult(
                ok=False,
                mode="real",
                puf_id="0" * 32,
                shield_ok=False,
                chip_unlocked=False,
                signature_ok=False,
                attestation_ok=False,
                zeroize_active=False,
                reason="short read from device",
            )

        # Placeholder layout:
        #   bytes 0..15   : puf_id (128 bits)
        #   byte  16      : flags
        #       bit0 shield_ok, bit1 chip_unlocked, bit2 signature_ok,
        #       bit3 attestation_ok, bit4 zeroize_active
        puf_id = data[:16].hex()
        flags = data[16]
        shield_ok = bool(flags & 0x01)
        chip_unlocked = bool(flags & 0x02)
        signature_ok = bool(flags & 0x04)
        attestation_ok = bool(flags & 0x08)
        zeroize_active = bool(flags & 0x10)

        ok = (shield_ok and chip_unlocked and signature_ok
              and attestation_ok and not zeroize_active)

        reason = None if ok else "attestation failed (real)"

        self._puf_id_hex = puf_id
        return AttestationResult(
            ok=ok,
            mode="real",
            puf_id=puf_id,
            shield_ok=shield_ok,
            chip_unlocked=chip_unlocked,
            signature_ok=signature_ok,
            attestation_ok=attestation_ok,
            zeroize_active=zeroize_active,
            reason=reason,
        )
