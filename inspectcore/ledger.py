"""
Ledger — append-only, hash-chained audit log for AI decisions.

Vendored minimal version of TurboLLM's qrap-lite
(https://github.com/karamik/TurboLLM — qrap-lite/), reduced to a pure
library: SQLite + SHA-256 hash chain + hybrid PQ signatures. No aiohttp,
no HTTP server, no Prometheus — those can be layered on top later.

Model
-----
Each "block" is a JSON record with:
    index          int, starting at 0
    ts             float, unix time
    payload        arbitrary JSON (the audit entry)
    prev_hash      hex SHA-256 of the previous block, or "0"*64 for genesis
    hash           hex SHA-256 over canonical_json(index, ts, payload, prev_hash)
    signature      ProofSignature dict (ECDSA + Dilithium3 hybrid)

The chain is tamper-evident: changing any field of any block breaks the
hash chain from that point forward, and `verify()` will report the first
bad index.

Storage
-------
Default path: ~/.inspectcore/ledger.sqlite
Override with `Ledger(path)` or `INSPECTCORE_LEDGER=/path/to/file.sqlite`.

Non-goals
---------
* Not a distributed ledger. Single file, single writer.
* Not a consensus system. No fees, no tokens, no wallets.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from .signer import Signer, ProofSignature


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GENESIS_PREV_HASH = "0" * 64
DEFAULT_DB = "~/.inspectcore/ledger.sqlite"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _canonical_json(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _compute_block_hash(index: int, ts: float, payload: Any,
                        prev_hash: str) -> str:
    body = {
        "index": index,
        "ts": round(float(ts), 6),
        "payload": payload,
        "prev_hash": prev_hash,
    }
    return _sha256_hex(_canonical_json(body))


# ---------------------------------------------------------------------------
# Block
# ---------------------------------------------------------------------------

@dataclass
class Block:
    index: int
    ts: float
    payload: Any
    prev_hash: str
    hash: str
    signature: Optional[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "ts": self.ts,
            "payload": self.payload,
            "prev_hash": self.prev_hash,
            "hash": self.hash,
            "signature": self.signature,
        }


# ---------------------------------------------------------------------------
# Ledger
# ---------------------------------------------------------------------------

class Ledger:
    """
    Append-only, hash-chained audit log backed by SQLite.

    Usage
    -----
        from inspectcore import Ledger, Signer

        led = Ledger()                     # ~/.inspectcore/ledger.sqlite
        signer = Signer()

        b = led.append({"decision": "APPROVED", "score": 0.12}, signer=signer)
        print(b.index, b.hash)

        ok, bad_index = led.verify(signer.public_keys())
        print(ok, bad_index)               # True None

        for blk in led.tail(5):
            print(blk.index, blk.payload)
    """

    def __init__(self, path: Optional[str] = None, *,
                 create: bool = True) -> None:
        if path is None:
            path = os.environ.get("INSPECTCORE_LEDGER", DEFAULT_DB)

        self.path = str(Path(os.path.expanduser(path)))
        if create:
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)

        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _init_schema(self) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS blocks (
                    idx         INTEGER PRIMARY KEY,
                    ts          REAL NOT NULL,
                    payload     TEXT NOT NULL,
                    prev_hash   TEXT NOT NULL,
                    hash        TEXT NOT NULL,
                    signature   TEXT
                )
                """
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_blocks_hash ON blocks(hash)"
            )

    # ------------------------------------------------------------------
    # Append
    # ------------------------------------------------------------------

    def append(self, payload: Any, *, signer: Optional[Signer] = None,
               ts: Optional[float] = None) -> Block:
        """
        Append a new block. Signs it with `signer` if provided.

        payload must be JSON-serialisable.
        """
        with self._lock:
            index = self._next_index()
            ts_val = float(ts) if ts is not None else time.time()
            prev_hash = self._last_hash()

            block_hash = _compute_block_hash(index, ts_val, payload, prev_hash)

            sig_dict: Optional[Dict[str, Any]] = None
            if signer is not None:
                # Sign a stable subset — do not include the signature itself.
                signed_body = {
                    "index": index,
                    "ts": round(ts_val, 6),
                    "payload": payload,
                    "prev_hash": prev_hash,
                    "hash": block_hash,
                }
                sig: ProofSignature = signer.sign(signed_body)
                sig_dict = sig.to_dict()

            with self._conn:
                self._conn.execute(
                    "INSERT INTO blocks(idx, ts, payload, prev_hash, hash, signature) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        index,
                        ts_val,
                        json.dumps(payload, sort_keys=True,
                                   separators=(",", ":"), ensure_ascii=False),
                        prev_hash,
                        block_hash,
                        json.dumps(sig_dict, sort_keys=True,
                                   separators=(",", ":")) if sig_dict else None,
                    ),
                )

            return Block(
                index=index,
                ts=ts_val,
                payload=payload,
                prev_hash=prev_hash,
                hash=block_hash,
                signature=sig_dict,
            )

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) AS n FROM blocks").fetchone()
            return int(row["n"])

    def get(self, index: int) -> Optional[Block]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM blocks WHERE idx = ?", (index,)
            ).fetchone()
        return self._row_to_block(row) if row else None

    def last(self) -> Optional[Block]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM blocks ORDER BY idx DESC LIMIT 1"
            ).fetchone()
        return self._row_to_block(row) if row else None

    def tail(self, n: int = 10) -> List[Block]:
        """Return the last `n` blocks in ascending order."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM blocks ORDER BY idx DESC LIMIT ?", (int(n),)
            ).fetchall()
        blocks = [self._row_to_block(r) for r in rows]
        blocks.reverse()
        return blocks

    def iter_all(self) -> Iterator[Block]:
        with self._lock:
            cur = self._conn.execute("SELECT * FROM blocks ORDER BY idx ASC")
            rows = cur.fetchall()
        for r in rows:
            yield self._row_to_block(r)

    # ------------------------------------------------------------------
    # Verify
    # ------------------------------------------------------------------

    def verify(self, pubkeys: Optional[Any] = None, *,
               check_signatures: bool = True) -> Dict[str, Any]:
        """
        Verify the whole chain.

        Returns a dict:
            {
              "ok": bool,
              "blocks": int,
              "first_bad_index": int | None,
              "reason": str | None,
              "signature_failures": [int, ...],
              "signature_checked": bool,
            }

        `pubkeys` is a PublicKeyBundle (from Signer.public_keys()).
        If None and check_signatures=True, signature verification is skipped
        and only the hash chain is validated.
        """
        prev = GENESIS_PREV_HASH
        n = 0
        first_bad: Optional[int] = None
        reason: Optional[str] = None
        sig_failures: List[int] = []

        verifier: Optional[Signer] = None
        if check_signatures and pubkeys is not None:
            # We need a Signer instance only for its verify() method.
            # It does not have to own the private key for ECDSA verification,
            # but simulated PQ mode requires the same instance that signed.
            # This is a known limitation of simulated PQ; the caller should
            # pass the signing Signer (not just the pubkeys) if PQ mode is
            # "simulated". We accept either a Signer or a PublicKeyBundle.
            if isinstance(pubkeys, Signer):
                verifier = pubkeys
                pubkeys = verifier.public_keys()

        for blk in self.iter_all():
            n += 1
            expected = _compute_block_hash(blk.index, blk.ts, blk.payload,
                                           blk.prev_hash)
            if expected != blk.hash:
                first_bad = blk.index
                reason = f"hash mismatch at block {blk.index}"
                break
            if blk.prev_hash != prev:
                first_bad = blk.index
                reason = f"prev_hash mismatch at block {blk.index}"
                break
            prev = blk.hash

            # Signature check (optional)
            if verifier is not None and blk.signature is not None:
                signed_body = {
                    "index": blk.index,
                    "ts": round(float(blk.ts), 6),
                    "payload": blk.payload,
                    "prev_hash": blk.prev_hash,
                    "hash": blk.hash,
                }
                try:
                    sig = ProofSignature.from_dict(blk.signature)
                    if not verifier.verify(signed_body, sig, pubkeys):
                        sig_failures.append(blk.index)
                except Exception:
                    sig_failures.append(blk.index)

        ok = (first_bad is None) and (not sig_failures)
        return {
            "ok": ok,
            "blocks": n,
            "first_bad_index": first_bad,
            "reason": reason,
            "signature_failures": sig_failures,
            "signature_checked": verifier is not None,
        }

    # ------------------------------------------------------------------
    # Admin
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Drop all blocks. Irreversible. For tests/dev only."""
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM blocks")

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def status(self) -> Dict[str, Any]:
        last = self.last()
        return {
            "path": self.path,
            "blocks": len(self),
            "last_hash": last.hash if last else None,
            "last_ts": last.ts if last else None,
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _next_index(self) -> int:
        row = self._conn.execute(
            "SELECT COALESCE(MAX(idx), -1) AS m FROM blocks"
        ).fetchone()
        return int(row["m"]) + 1

    def _last_hash(self) -> str:
        row = self._conn.execute(
            "SELECT hash FROM blocks ORDER BY idx DESC LIMIT 1"
        ).fetchone()
        return row["hash"] if row else GENESIS_PREV_HASH

    @staticmethod
    def _row_to_block(row: sqlite3.Row) -> Block:
        sig = None
        if row["signature"]:
            try:
                sig = json.loads(row["signature"])
            except Exception:
                sig = None
        try:
            payload = json.loads(row["payload"])
        except Exception:
            payload = row["payload"]
        return Block(
            index=int(row["idx"]),
            ts=float(row["ts"]),
            payload=payload,
            prev_hash=row["prev_hash"],
            hash=row["hash"],
            signature=sig,
        )

    # ------------------------------------------------------------------
    # Context manager sugar
    # ------------------------------------------------------------------

    def __enter__(self) -> "Ledger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
