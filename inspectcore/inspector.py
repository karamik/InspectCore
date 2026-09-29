"""
G-Space Inspector — real-time analysis of hidden activations.

Extracted and simplified from TurboLLM's G-Space Inspector
(https://github.com/karamik/TurboLLM — g_inspector/, real_inspector.py,
advanced_gspace_inspector.py) and TOTAL-Neuro's "immutable anchor vector"
idea.

What it does
------------
Given a vector (or tensor) of hidden activations from a language model,
it computes a small set of interpretable features and returns a decision:

    APPROVED   — activations look "normal", model output is trusted
    REFLECTED  — mild anomaly; caller should re-prompt / re-run
    BLOCKED    — strong anomaly; caller must refuse

Features (all numpy, no torch, no scipy):
    * spectral_entropy     — FFT-based, how spread is power across freqs
    * spectral_centroid    — "center of mass" of the power spectrum
    * cosine_drift         — 1 - cos(vec, adaptive_reference)
    * norm_ratio           — |vec| / |reference|
    * mean_abs             — mean(|vec|), cheap sanity signal

The adaptive reference is an EMA of vectors that were APPROVED.
It moves, but slowly, and only after a trusted decision was made.

No ML classifier yet. No training. Deterministic, ~<1 ms per call for
typical hidden sizes (D <= 8192).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Decision constants
# ---------------------------------------------------------------------------

DECISION_APPROVED = "APPROVED"
DECISION_REFLECTED = "REFLECTED"
DECISION_BLOCKED = "BLOCKED"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class InspectorConfig:
    """Tunable thresholds. All anomaly values are in [0, 1]."""

    # Below approve_threshold -> APPROVED
    approve_threshold: float = 0.25
    # Between approve and block -> REFLECTED
    # Above block_threshold -> BLOCKED
    block_threshold: float = 0.65

    # How many samples before the reference vector is considered "warm".
    warmup_samples: int = 3

    # EMA smoothing for the adaptive reference.
    ema_alpha: float = 0.1

    # Cap on the number of recent metrics kept in memory (for state()).
    max_history: int = 100


# ---------------------------------------------------------------------------
# Inspector
# ---------------------------------------------------------------------------

class Inspector:
    """
    Analyse hidden activations and return a safety decision.

    Usage
    -----
        from inspectcore import Inspector
        import numpy as np

        insp = Inspector()
        result = insp.inspect(np.random.randn(4096).astype(np.float32))
        print(result["decision"], result["anomaly_score"])
    """

    def __init__(self, config: Optional[InspectorConfig] = None) -> None:
        self.config = config or InspectorConfig()

        self._reference: Optional[np.ndarray] = None
        self._ref_entropy: Optional[float] = None
        self._ref_norm: Optional[float] = None
        self._ref_mean_abs: Optional[float] = None

        self._seen: int = 0
        self._approved: int = 0
        self._reflected: int = 0
        self._blocked: int = 0

        self._history: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def inspect(self, hidden_states: Any) -> Dict[str, Any]:
        """
        Run inspection on a batch/vector of hidden states.

        Parameters
        ----------
        hidden_states : array-like
            Any of:
              - (D,)          single vector
              - (L, D)        per-layer vectors, mean-pooled over L
              - (B, L, D)     batched, mean-pooled over B and L

        Returns
        -------
        dict with keys:
            decision, anomaly_score, confidence, warmup,
            metrics, reference_seen, latency_ms
        """
        t0 = time.perf_counter()

        vec = self._to_vector(hidden_states)

        metrics = self._compute_metrics(vec)
        anomaly = self._anomaly(metrics)

        warmup = self._reference is None or self._seen < self.config.warmup_samples

        decision = self._decide(anomaly, warmup)
        confidence = self._confidence(anomaly, decision, warmup)

        # Update reference only on APPROVED decisions.
        if decision == DECISION_APPROVED:
            self._update_reference(vec, metrics)
            self._approved += 1
        elif decision == DECISION_REFLECTED:
            self._reflected += 1
        else:
            self._blocked += 1

        self._seen += 1

        latency_ms = (time.perf_counter() - t0) * 1000.0

        result: Dict[str, Any] = {
            "decision": decision,
            "anomaly_score": float(anomaly),
            "confidence": float(confidence),
            "warmup": bool(warmup),
            "metrics": metrics,
            "reference_seen": self._seen,
            "latency_ms": float(latency_ms),
        }

        self._history.append(
            {
                "ts": time.time(),
                "decision": decision,
                "anomaly_score": float(anomaly),
                "metrics": metrics,
            }
        )
        if len(self._history) > self.config.max_history:
            self._history = self._history[-self.config.max_history:]

        return result

    def explain(self, result: Dict[str, Any]) -> str:
        """Human-readable one-liner for a result dict."""
        m = result["metrics"]
        return (
            f"{result['decision']} "
            f"(anomaly={result['anomaly_score']:.3f}, "
            f"drift={m['cosine_drift']:.3f}, "
            f"entropy={m['spectral_entropy']:.3f}, "
            f"norm_ratio={m['norm_ratio']:.3f})"
        )

    def reset(self) -> None:
        """Forget the reference vector and all counters."""
        self._reference = None
        self._ref_entropy = None
        self._ref_norm = None
        self._ref_mean_abs = None
        self._seen = 0
        self._approved = 0
        self._reflected = 0
        self._blocked = 0
        self._history = []

    def state(self) -> Dict[str, Any]:
        """Serialisable snapshot (no numpy arrays)."""
        return {
            "seen": self._seen,
            "approved": self._approved,
            "reflected": self._reflected,
            "blocked": self._blocked,
            "has_reference": self._reference is not None,
            "ref_entropy": self._ref_entropy,
            "ref_norm": self._ref_norm,
            "ref_mean_abs": self._ref_mean_abs,
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _to_vector(hidden_states: Any) -> np.ndarray:
        """
        Normalise arbitrary shapes to a single 1-D float32 vector.

        Uses mean-pooling over all leading dimensions.
        """
        arr = np.asarray(hidden_states, dtype=np.float32)

        if arr.ndim == 0:
            raise ValueError("hidden_states must have at least 1 dimension")

        if arr.ndim == 1:
            vec = arr
        else:
            # mean over all but the last dimension
            axes = tuple(range(arr.ndim - 1))
            vec = arr.mean(axis=axes)

        if vec.size == 0:
            raise ValueError("hidden_states is empty")

        # Replace NaN/Inf with 0 to keep metrics finite.
        vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)

        return vec.astype(np.float32, copy=False)

    @staticmethod
    def _spectral_features(vec: np.ndarray) -> Dict[str, float]:
        """
        FFT-based features.

        spectral_entropy   — Shannon entropy of the normalised power spectrum.
                             0 = all power at one frequency, 1 = flat.
        spectral_centroid  — weighted mean frequency, in [0, 1] of Nyquist.
        """
        n = vec.size
        if n < 4:
            return {"spectral_entropy": 0.0, "spectral_centroid": 0.0}

        # Remove DC before FFT so the mean does not dominate.
        x = vec - vec.mean()
        spectrum = np.fft.rfft(x)
        power = (np.abs(spectrum) ** 2).astype(np.float64)

        total = power.sum()
        if total <= 0.0:
            return {"spectral_entropy": 0.0, "spectral_centroid": 0.0}

        p = power / total

        # Shannon entropy, normalised by log(number of bins).
        eps = 1e-12
        h = -np.sum(p * np.log(p + eps))
        h_max = math.log(p.size) if p.size > 1 else 1.0
        spectral_entropy = float(h / h_max) if h_max > 0 else 0.0

        # Centroid: weighted mean of bin indices, normalised to [0, 1].
        idx = np.arange(p.size, dtype=np.float64)
        centroid_raw = float(np.sum(p * idx))
        spectral_centroid = float(centroid_raw / max(p.size - 1, 1))

        return {
            "spectral_entropy": spectral_entropy,
            "spectral_centroid": spectral_centroid,
        }

    def _compute_metrics(self, vec: np.ndarray) -> Dict[str, float]:
        spec = self._spectral_features(vec)
        norm = float(np.linalg.norm(vec))
        mean_abs = float(np.mean(np.abs(vec)))

        if self._reference is None:
            drift = 0.0
        else:
            ref = self._reference
            # Pad/truncate to common length if needed (defensive).
            if ref.shape != vec.shape:
                m = min(ref.size, vec.size)
                a = vec[:m]
                b = ref[:m]
            else:
                a, b = vec, ref
            na = float(np.linalg.norm(a))
            nb = float(np.linalg.norm(b))
            if na < 1e-12 or nb < 1e-12:
                drift = 1.0
            else:
                cos = float(np.dot(a, b) / (na * nb))
                cos = max(-1.0, min(1.0, cos))
                drift = 1.0 - cos

        ref_norm = self._ref_norm if self._ref_norm else norm
        norm_ratio = float(norm / ref_norm) if ref_norm > 1e-12 else 1.0

        return {
            "spectral_entropy": spec["spectral_entropy"],
            "spectral_centroid": spec["spectral_centroid"],
            "cosine_drift": float(drift),
            "norm": norm,
            "norm_ratio": norm_ratio,
            "mean_abs": mean_abs,
        }

    def _anomaly(self, m: Dict[str, float]) -> float:
        """
        Combine features into a single anomaly score in [0, 1].

        Weights are heuristic, tunable in InspectorConfig later.
        """
        drift_term = min(max(m["cosine_drift"], 0.0), 1.0)

        if self._ref_entropy is None:
            entropy_term = 0.0
        else:
            denom = max(self._ref_entropy, 1e-6)
            entropy_term = min(abs(m["spectral_entropy"] - self._ref_entropy) / denom, 1.0)

        norm_term = min(abs(math.log(max(m["norm_ratio"], 1e-6))), 1.0)

        if self._ref_mean_abs is None or self._ref_mean_abs < 1e-12:
            mean_term = 0.0
        else:
            mean_term = min(abs(m["mean_abs"] - self._ref_mean_abs) / self._ref_mean_abs, 1.0)

        score = (
            0.55 * drift_term
            + 0.20 * entropy_term
            + 0.15 * norm_term
            + 0.10 * mean_term
        )
        return float(min(max(score, 0.0), 1.0))

    def _decide(self, anomaly: float, warmup: bool) -> str:
        if warmup:
            # Not enough baseline yet — trust, but flag it as warmup.
            return DECISION_APPROVED
        if anomaly < self.config.approve_threshold:
            return DECISION_APPROVED
        if anomaly < self.config.block_threshold:
            return DECISION_REFLECTED
        return DECISION_BLOCKED

    @staticmethod
    def _confidence(anomaly: float, decision: str, warmup: bool) -> float:
        """
        Distance from the nearest decision boundary, in [0, 1].
        1.0 = very confident, 0.0 = right on the edge.
        """
        if warmup:
            return 0.3
        # Boundary distance assuming thresholds 0.25 / 0.65 baked here;
        # kept simple on purpose. Not used for the decision itself.
        if decision == DECISION_APPROVED:
            return float(min(max((0.25 - anomaly) / 0.25, 0.0), 1.0)) if anomaly < 0.25 else 0.0
        if decision == DECISION_BLOCKED:
            return float(min(max((anomaly - 0.65) / 0.35, 0.0), 1.0)) if anomaly > 0.65 else 0.0
        # REFLECTED
        return float(min(abs(anomaly - 0.45) / 0.20, 1.0))

    def _update_reference(self, vec: np.ndarray, m: Dict[str, float]) -> None:
        a = self.config.ema_alpha

        if self._reference is None:
            self._reference = vec.astype(np.float32, copy=True)
            self._ref_entropy = m["spectral_entropy"]
            self._ref_norm = m["norm"] if m["norm"] > 1e-12 else 1.0
            self._ref_mean_abs = m["mean_abs"] if m["mean_abs"] > 1e-12 else 1e-6
            return

        # Pad/truncate to common size defensively.
        ref = self._reference
        if ref.shape != vec.shape:
            m_sz = min(ref.size, vec.size)
            ref = ref[:m_sz]
            vec = vec[:m_sz]

        self._reference = ((1.0 - a) * ref + a * vec).astype(np.float32, copy=False)

        self._ref_entropy = (1.0 - a) * self._ref_entropy + a * m["spectral_entropy"]
        self._ref_norm = (1.0 - a) * self._ref_norm + a * max(m["norm"], 1e-12)
        self._ref_mean_abs = (1.0 - a) * self._ref_mean_abs + a * max(m["mean_abs"], 1e-12)


# ---------------------------------------------------------------------------
# Synthetic helper (used by tests / CLI demo)
# ---------------------------------------------------------------------------

def synthetic_hidden_states(dim: int = 512, seed: Optional[int] = None,
                            scale: float = 1.0) -> np.ndarray:
    """Generate a reproducible random vector for demos and tests."""
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(dim).astype(np.float32) * scale)
